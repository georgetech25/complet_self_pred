import streamlit as st
import joblib

# Set up page configuration
st.set_page_config(
    page_icon="logo.jpeg",
    page_title="Self Diagnosis",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("PAGE NAVIGATION")
selected_page = st.sidebar.radio("PAGES", ["Children Prediction", "Adult Prediction"])

# Load models and vectorizers for both children and adults
children_model = joblib.load('infant_sickness_model.pkl')
children_vectorizer = joblib.load('infant_symptoms_vectorizer.pkl')

adult_model = joblib.load('sickness_predictor_model.pkl')
adult_vectorizer = joblib.load('symptoms_vectorizer.pkl')

# Mappings for prescription, causes, and symptoms (all entries added)
def load_mappings():
    return {
        "children": {
            "prescriptions": {
                "Teething Fever": "Cool compress, hydration, teething gel",
                "Common Cold": "Saline drops, humidifier, rest",
                "Stomach Flu": "Oral rehydration salts, hydration",
                "Roseola": "Acetaminophen, hydration, see doctor if needed",
                "Ear Infection": "Pain relief drops, antibiotics for infection",
                "Bronchiolitis": "Nebulizer, consult doctor",
                "Infant Colic": "Burping after meals, massage tummy",
                "Gastroesophageal Reflux": "Small frequent feeds, consult doctor",
                "Conjunctivitis": "Clean eyes with warm water, antibiotic drops",
                "Measles": "Acetaminophen, hydration, see doctor",
                "Tonsillitis": "Pain relievers, hydration, consult doctor",
                "Brain Tumor": "Surgery, radiation, chemotherapy",
                "Stroke": "Emergency care, physical therapy, blood thinners",
                "Multiple Sclerosis": "Disease-modifying drugs, physical therapy",
                "Meningitis": "Antibiotics, fluids, rest",
                "Kidney Infection": "Antibiotics, pain relief, hydration",
                "Food Poisoning": "Hydration, rest, anti-nausea medication",
                "Gastritis": "Antacids, small meals, avoid spicy food",
                "Hepatitis": "Rest, hydration, antiviral medication",
                "Lymphoma": "Chemotherapy, radiotherapy, consultation with oncologist",
                "Tuberculosis": "Antibiotics, isolation, follow-up with doctor",
                "Allergic Reaction": "Antihistamines, epinephrine if severe"
            },
            "causes": {
                "Teething Fever": "Teething process causing gum inflammation",
                "Common Cold": "Viral infection, exposure to infected individuals",
                "Stomach Flu": "Viral gastroenteritis, contaminated food or water",
                "Roseola": "Human herpesvirus 6 or 7",
                "Ear Infection": "Bacterial or viral infection in the middle ear",
                "Bronchiolitis": "RSV infection, common in infants during winter",
                "Infant Colic": "Digestive issues, overfeeding, or air swallowed during feeding",
                "Gastroesophageal Reflux": "Immature digestive system causing acid reflux",
                "Conjunctivitis": "Bacterial or viral infection, allergens, or irritants",
                "Measles": "Measles virus, lack of vaccination",
                "Tonsillitis": "Bacterial or viral infection of the tonsils",
                "Brain Tumor": "Abnormal cell growth, genetic factors",
                "Stroke": "Blocked blood flow to the brain, burst blood vessels",
                "Multiple Sclerosis": "Immune system attacks the central nervous system, genetic factors",
                "Meningitis": "Bacterial or viral infection affecting brain membranes",
                "Kidney Infection": "Bacterial infection, untreated UTI",
                "Food Poisoning": "Contaminated food or water, bacterial toxins",
                "Gastritis": "Helicobacter pylori infection, overuse of NSAIDs, spicy foods",
                "Hepatitis": "Viral infections (A, B, C), alcohol abuse, toxins",
                "Lymphoma": "Abnormal lymphocyte growth, genetic factors",
                "Tuberculosis": "Mycobacterium tuberculosis infection, airborne transmission",
                "Allergic Reaction": "Exposure to allergens (food, pollen, medications)"
            },
            "symptoms": {
                "Teething Fever": "Mild fever, drooling, irritability",
                "Common Cold": "Runny nose, congestion, sneezing, mild fever",
                "Stomach Flu": "Vomiting, diarrhea, abdominal cramps",
                "Roseola": "High fever followed by rash, irritability",
                "Ear Infection": "Ear pain, fever, difficulty sleeping",
                "Bronchiolitis": "Wheezing, rapid breathing, coughing",
                "Infant Colic": "Prolonged crying, clenching fists, gas",
                "Gastroesophageal Reflux": "Frequent spit-ups, irritability during feeds",
                "Conjunctivitis": "Red eyes, discharge, tearing",
                "Measles": "Fever, rash, cough, runny nose",
                "Tonsillitis": "Sore throat, swollen tonsils, fever",
                "Brain Tumor": "Headache, nausea, vision problems",
                "Stroke": "Weakness on one side, trouble speaking, dizziness",
                "Multiple Sclerosis": "Muscle weakness, numbness, coordination issues",
                "Meningitis": "Stiff neck, fever, headache, sensitivity to light",
                "Kidney Infection": "Fever, back pain, painful urination",
                "Food Poisoning": "Nausea, vomiting, diarrhea, stomach pain",
                "Gastritis": "Stomach pain, nausea, bloating",
                "Hepatitis": "Jaundice, fatigue, abdominal pain",
                "Lymphoma": "Swollen lymph nodes, fatigue, weight loss",
                "Tuberculosis": "Persistent cough, night sweats, weight loss",
                "Allergic Reaction": "Itching, swelling, difficulty breathing"
            }
        },
        "adult": {
            "prescriptions": {
                "Flu": "Rest, hydration, Paracetamol",
                "Migraine": "Painkillers, rest in a dark room",
                "Common Cold": "Antihistamines, warm water gargle",
                "Pneumonia": "Antibiotics, oxygen therapy",
                "Asthma": "Inhalers, avoid triggers, bronchodilators",
                "COVID-19": "Isolation, antiviral medications, rest, hydration",
                "Dengue Fever": "Paracetamol, fluid replacement",
                "Brain Tumor": "Surgical intervention, chemotherapy, radiation therapy",
                "Stroke": "Emergency medical care, blood thinners, physical therapy",
                "Allergic Reaction": "Antihistamines, epinephrine (for severe reactions), avoid allergens",
                "Tonsillitis": "Warm saltwater gargle, antibiotics (if bacterial), pain relievers",
                "Kidney Infection": "Antibiotics, hydration, pain relievers",
                "Lymphoma": "Chemotherapy, radiation therapy, targeted therapy",
                "Malaria": "Antimalarial drugs, hydration, fever management",
                "Tuberculosis": "Antitubercular drugs, long-term treatment regimen",
                "Food Poisoning": "Rehydration, rest, anti-nausea medications",
                "Multiple Sclerosis": "Immunomodulatory drugs, physical therapy, symptom management",
                "Gastritis": "Antacids, avoid spicy foods, proton pump inhibitors",
                "Meningitis": "Antibiotics (if bacterial), corticosteroids, hospitalization",
                "Hepatitis": "Rest, antiviral medications, avoid alcohol, liver support therapy"
            },
            "causes": {
                "Flu": "Influenza virus, seasonal changes, weakened immunity",
                "Migraine": "Stress, lack of sleep, certain foods or smells",
                "Common Cold": "Rhinovirus, exposure to cold weather, low immunity",
                "Pneumonia": "Bacterial infection, viral infection, fungal infection",
                "Asthma": "Allergic reactions, environmental triggers, genetic factors",
                "COVID-19": "Coronavirus infection, close contact with infected persons",
                "Dengue Fever": "Mosquito bites (Aedes aegypti), tropical climates",
                "Brain Tumor": "Genetic mutations, radiation exposure",
                "Stroke": "Blocked blood flow to the brain, burst blood vessels",
                "Allergic Reaction": "Exposure to allergens (pollen, food, medication)",
                "Tonsillitis": "Bacterial or viral infection of the tonsils",
                "Kidney Infection": "Bacterial infection, untreated urinary tract infections",
                "Lymphoma": "Abnormal growth of lymphocytes, genetic predisposition",
                "Malaria": "Plasmodium parasite through mosquito bites",
                "Tuberculosis": "Mycobacterium tuberculosis infection, airborne transmission",
                "Food Poisoning": "Contaminated food or water, bacterial toxins",
                "Multiple Sclerosis": "Immune system attacks the central nervous system, genetic factors",
                "Gastritis": "Helicobacter pylori infection, overuse of NSAIDs, alcohol",
                "Meningitis": "Infection of the protective membranes of the brain, bacterial or viral",
                "Hepatitis": "Viral infections (Hepatitis A, B, C, etc.), alcohol abuse, certain medications, toxins"
            },
            "symptoms": {
                "Flu": "Fever, cough, sore throat, muscle aches",
                "Migraine": "Severe headache, nausea, sensitivity to light",
                "Common Cold": "Runny nose, congestion, sneezing, mild fever",
                "Pneumonia": "Cough with phlegm, chest pain, shortness of breath",
                "Asthma": "Wheezing, shortness of breath, chest tightness",
                "COVID-19": "Fever, cough, loss of taste or smell, fatigue",
                "Dengue Fever": "High fever, severe headache, retro-orbital pain",
                "Brain Tumor": "Headache, nausea, vision problems, seizures",
                "Stroke": "Sudden numbness, confusion, difficulty speaking, dizziness",
                "Allergic Reaction": "Itching, swelling, difficulty breathing, rash",
                "Tonsillitis": "Sore throat, swollen tonsils, fever",
                "Kidney Infection": "Fever, pain in back or side, painful urination",
                "Lymphoma": "Swollen lymph nodes, fever, night sweats, weight loss",
                "Malaria": "Fever, chills, sweating, headache",
                "Tuberculosis": "Persistent cough, night sweats, weight loss, fatigue",
                "Food Poisoning": "Nausea, vomiting, diarrhea, stomach cramps",
                "Multiple Sclerosis": "Fatigue, numbness, difficulty walking, muscle weakness",
                "Gastritis": "Stomach pain, bloating, nausea, indigestion",
                "Meningitis": "Stiff neck, fever, headache, nausea",
                "Hepatitis": "Fatigue, jaundice, abdominal pain, nausea"
            }
        }
    }

mappings = load_mappings()

# Function for sickness prediction
def predict_sickness(model, vectorizer, symptoms, category):
    symptoms_vec = vectorizer.transform([symptoms])
    sickness = model.predict(symptoms_vec)[0]
    prescription = mappings[category]["prescriptions"].get(sickness, "Consult a doctor.")
    causes = mappings[category]["causes"].get(sickness, "Unknown causes. Consult a doctor.")
    related_symptoms = mappings[category]["symptoms"].get(sickness, "No specific symptoms listed.")
    return sickness, prescription, causes, related_symptoms

# Display corresponding page content
if selected_page == "Children Prediction":
    st.title("Infant Sickness Prediction App")
    st.write("Enter the symptoms of the infant to get a diagnosis, prescription, possible causes, and related symptoms.")
    st.image("bg-removebg-preview.png", width=300)

    # Get possible symptoms from vectorizer
    all_symptoms = children_vectorizer.get_feature_names_out()
    st.subheader("Possible Symptoms: ")
    st.write(", ".join(all_symptoms))

    # User input for symptoms
    symptoms = st.text_area("Symptoms (e.g., fever, cough, fatigue):")

    if st.button("Predict"):
        if symptoms.strip():
            sickness, prescription, causes, related_symptoms = predict_sickness(
                children_model, children_vectorizer, symptoms, "children"
            )
            st.subheader(f"Predicted Sickness: {sickness}")
            st.write(f"Prescription: {prescription}")
            st.write(f"Possible Causes: {causes}")
            st.write(f"Related Symptoms: {related_symptoms}")
        else:
            st.error("Please enter symptoms.")

elif selected_page == "Adult Prediction":
    st.title("Adult Sickness Diagnosis App")
    st.write("Enter your symptoms to get a diagnosis, prescription, possible causes, and related symptoms.")
    st.image("testoscope.jpeg", width=200)

    # Get possible symptoms from vectorizer
    all_symptoms = adult_vectorizer.get_feature_names_out()
    st.subheader("Possible Symptoms: ")
    st.write(", ".join(all_symptoms))

    # User input for symptoms
    symptoms = st.text_area("Symptoms (e.g., fever, cough, fatigue):")

    if st.button("Predict"):
        if symptoms.strip():
            sickness, prescription, causes, related_symptoms = predict_sickness(
                adult_model, adult_vectorizer, symptoms, "adult"
            )
            st.subheader(f"Predicted Sickness: {sickness}")
            st.write(f"Prescription: {prescription}")
            st.write(f"Possible Causes: {causes}")
            st.write(f"Related Symptoms: {related_symptoms}")
        else:
            st.error("Please enter symptoms.")
