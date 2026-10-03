from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, OWL, RDFS, XSD

# Define Namespace
CLIN = Namespace("http://example.org/clinical#")

def create_clinical_graph():
    g = Graph()
    g.bind("clin", CLIN)
    g.bind("owl", OWL)
    
    # Define Ontology (Classes)
    g.add((CLIN.ClinicalTrial, RDF.type, OWL.Class))
    g.add((CLIN.Condition, RDF.type, OWL.Class))
    g.add((CLIN.Drug, RDF.type, OWL.Class))
    g.add((CLIN.Phase, RDF.type, OWL.Class))
    g.add((CLIN.Outcome, RDF.type, OWL.Class))
    
    # Define Object Properties
    g.add((CLIN.hasIntervention, RDF.type, OWL.ObjectProperty))
    g.add((CLIN.studiesCondition, RDF.type, OWL.ObjectProperty))
    g.add((CLIN.hasPhase, RDF.type, OWL.ObjectProperty))
    g.add((CLIN.hasOutcome, RDF.type, OWL.ObjectProperty))

    # Add Synthetic Data
    trials_data = [
        ("TrialA", "Metformin", "Type 2 Diabetes", "Phase 3", "HbA1c Reduction"),
        ("TrialB", "Insulin Glargine", "Type 1 Diabetes", "Phase 4", "Fasting Glucose"),
        ("TrialC", "Pembrolizumab", "Melanoma", "Phase 3", "Overall Survival"),
        ("TrialD", "Nivolumab", "Lung Cancer", "Phase 2", "Progression Free Survival"),
        ("TrialE", "Atorvastatin", "Hyperlipidemia", "Phase 3", "LDL Cholesterol Reduction"),
        ("TrialF", "Lisinopril", "Hypertension", "Phase 4", "Blood Pressure Reduction"),
        ("TrialG", "Ocrelizumab", "Multiple Sclerosis", "Phase 3", "Annualized Relapse Rate"),
        ("TrialH", "Aducanumab", "Alzheimer's Disease", "Phase 3", "Cognitive Decline Delay"),
        ("TrialI", "Semaglutide", "Obesity", "Phase 3", "Weight Loss"),
        ("TrialJ", "Rivaroxaban", "Deep Vein Thrombosis", "Phase 3", "Recurrent VTE Prevention")
    ]
    
    for trial_id, drug_name, condition_name, phase_name, outcome_name in trials_data:
        trial_uri = CLIN[trial_id]
        drug_uri = CLIN[drug_name.replace(" ", "_").replace("'", "")]
        cond_uri = CLIN[condition_name.replace(" ", "_").replace("'", "")]
        phase_uri = CLIN[phase_name.replace(" ", "_").replace("'", "")]
        outcome_uri = CLIN[outcome_name.replace(" ", "_").replace("'", "")]
        
        # Types
        g.add((trial_uri, RDF.type, CLIN.ClinicalTrial))
        g.add((trial_uri, RDFS.label, Literal(f"Clinical Trial {trial_id[-1]}")))
        
        g.add((drug_uri, RDF.type, CLIN.Drug))
        g.add((drug_uri, RDFS.label, Literal(drug_name)))
        
        g.add((cond_uri, RDF.type, CLIN.Condition))
        g.add((cond_uri, RDFS.label, Literal(condition_name)))
        
        g.add((phase_uri, RDF.type, CLIN.Phase))
        g.add((phase_uri, RDFS.label, Literal(phase_name)))
        
        g.add((outcome_uri, RDF.type, CLIN.Outcome))
        g.add((outcome_uri, RDFS.label, Literal(outcome_name)))
        
        # Relationships
        g.add((trial_uri, CLIN.hasIntervention, drug_uri))
        g.add((trial_uri, CLIN.studiesCondition, cond_uri))
        g.add((trial_uri, CLIN.hasPhase, phase_uri))
        g.add((trial_uri, CLIN.hasOutcome, outcome_uri))
        
    return g

# Initialize singleton graph
clinical_graph = create_clinical_graph()

def execute_sparql(query: str) -> str:
    """
    Executes a SPARQL query on the clinical graph and returns a formatted string.
    """
    try:
        results = clinical_graph.query(query)
        output = []
        for row in results:
            row_dict = row.asdict()
            row_str = ", ".join(f"{k}: {v}" for k, v in row_dict.items())
            output.append(row_str)
        if not output:
            return "No matching facts found in the Clinical Knowledge Graph."
        return "\n".join(output)
    except Exception as e:
        return f"SPARQL Query Error: {str(e)}"
