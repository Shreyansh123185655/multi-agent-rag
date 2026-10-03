from typing import List
from typing_extensions import TypedDict
import re

import os
os.environ.setdefault("USER_AGENT", "LangGraph-RAG-Agent/1.0")

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from clinical_kg import execute_sparql

class GraphState(TypedDict):
    question:   str
    generation: str
    documents:  List[Document]
    source:     str


def build_graph(retriever, wiki_tool, question_router, groq_api_key: str):

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="qwen/qwen3.8-27b",
    )

    # ── Nodes ────────────────────────────────────────────────

    def retrieve(state: GraphState) -> dict:
        docs = retriever.invoke(state["question"]) or []
        return {
            "documents": docs,
            "question":  state["question"],
            "source":    "vectorstore",
        }

    def knowledge_graph(state: GraphState) -> dict:
        sparql_prompt = (
            "You are an expert SPARQL developer for clinical trials.\n"
            "Generate a SPARQL query to answer the user's question.\n"
            "Ontology Classes: clin:ClinicalTrial, clin:Drug, clin:Condition, clin:Phase, clin:Outcome.\n"
            "Ontology Properties: clin:hasIntervention, clin:studiesCondition, clin:hasPhase, clin:hasOutcome.\n"
            "Labels: All instances have string labels attached via rdfs:label.\n"
            "Prefixes to include:\n"
            "PREFIX clin: <http://example.org/clinical#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "Return ONLY the raw SPARQL query, no markdown fences or explanation.\n\n"
            f"Question: {state['question']}\nSPARQL:"
        )
        query = llm.invoke(sparql_prompt).content.strip()
        query = query.strip("`").removeprefix("sparql\n").strip()
        
        facts = execute_sparql(query)
        
        # Hybrid RAG: retrieve unstructured docs as well
        docs = retriever.invoke(state["question"]) or []
        kg_doc = Document(page_content=f"Structured Knowledge Graph Facts:\n{facts}")
        docs.append(kg_doc)
        
        return {
            "documents": docs,
            "question":  state["question"],
            "source":    "knowledge_graph",
        }

    def wiki_search(state: GraphState) -> dict:
        try:
            docs = wiki_tool.invoke({"query": state["question"]}) or ""
        except Exception:
            prompt = (
                "Answer this factual question directly and concisely. "
                "If there are multiple accepted contributors, mention the main names.\n\n"
                f"Question: {state['question']}\nAnswer:"
            )
            response = llm.invoke(prompt)
            return {
                "documents":  [],
                "question":   state["question"],
                "generation": response.content,
                "source":     "general_chat",
            }

        return {
            "documents": [Document(page_content=docs)],
            "question":  state["question"],
            "source":    "wiki_search",
        }

    def general_chat(state: GraphState) -> dict:
        """Handle greetings, small talk, and casual conversation directly."""
        prompt = (
            "You are a warm, friendly, and helpful AI assistant. "
            "Respond naturally and concisely to the user's message. "
            "Be engaging and personable — no need to look anything up.\n\n"
            f"User: {state['question']}\nAssistant:"
        )
        response = llm.invoke(prompt)
        return {
            "documents":  [],
            "question":   state["question"],
            "generation": response.content,
            "source":     "general_chat",
        }

    def generate(state: GraphState) -> dict:
        if state.get("generation"):
            return state

        context = "\n\n".join(doc.page_content for doc in state["documents"])
        if state["source"] == "wiki_search":
            prompt = (
                "You are a precise factual assistant.\n"
                "Use the Wikipedia context when it answers the question. "
                "If the Wikipedia context is irrelevant or too narrow, answer from general knowledge instead. "
                "Keep the answer concise and do not mention the context.\n\n"
                f"Wikipedia context:\n{context}\n\n"
                f"Question: {state['question']}\nAnswer:"
            )
        elif state["source"] == "knowledge_graph":
            prompt = (
                "You are a precise clinical AI assistant.\n"
                "Answer using the provided structured knowledge graph facts and unstructured documents.\n"
                "Synthesize the best answer and do not invent clinical facts.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {state['question']}\nAnswer:"
            )
        else:
            prompt = (
                "You are a precise and helpful AI assistant.\n"
                "Answer using the provided knowledge-base context.\n"
                "Synthesize the best answer from relevant passages.\n"
                "If the context truly does not contain the answer, say you don't know.\n\n"
                f"Knowledge-base context:\n{context}\n\n"
                f"Question: {state['question']}\nAnswer:"
            )
        response = llm.invoke(prompt)
        return {
            "documents":  state["documents"],
            "question":   state["question"],
            "generation": response.content,
            "source":     state["source"],
        }

    def route_question(state: GraphState) -> str:
        question = state["question"].strip()
        if re.fullmatch(r"(?i)(hi|hello|hey|thanks|thank you|bye|goodbye|good morning|good afternoon|good evening)[!. ]*", question):
            return "general_chat"

        try:
            result = question_router.invoke({"question": question})
        except Exception:
            return "wiki_search"

        datasource = getattr(result, "datasource", None)
        if datasource in {"vectorstore", "knowledge_graph", "wiki_search", "general_chat"}:
            return datasource
        return "wiki_search"

    # ── Graph ────────────────────────────────────────────────

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve",        retrieve)
    workflow.add_node("knowledge_graph", knowledge_graph)
    workflow.add_node("wiki_search",     wiki_search)
    workflow.add_node("generate",        generate)
    workflow.add_node("general_chat",    general_chat)

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "vectorstore":     "retrieve",
            "knowledge_graph": "knowledge_graph",
            "wiki_search":     "wiki_search",
            "general_chat":    "general_chat",
        },
    )

    workflow.add_edge("retrieve",        "generate")
    workflow.add_edge("knowledge_graph", "generate")
    workflow.add_edge("wiki_search",     "generate")
    workflow.add_edge("general_chat",    END)          # no generate step needed
    workflow.add_edge("generate",        END)

    return workflow.compile()
