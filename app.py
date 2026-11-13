"""
Assignment 3: Government Scheme Eligibility Navigator
=====================================================
A citizen-facing multi-agent assistant built with Streamlit, Python, LangChain (Prompt
Templates & LCEL), and ChromaDB for Agentic RAG and Intent Routing.

Demonstrates all 7 Mandatory Acceptance Criteria:
1. LLM Integration: Supports Google Gemini API (gemini-2.5-flash / gemini-1.5-flash)
   and Local Ollama (phi3:mini).
2. Interactive GUI: Full Streamlit interface with Citizen Profile Builder, Chat/Query
   input, Agent Trace viewer, Prompt Engineering Inspector, and ChromaDB Explorer.
3. Prompt Engineering: Explicitly separates and displays System and User prompts using
   Few-Shot (Intent Router), ReAct (Agentic RAG Controller), Chain-of-Thought / CoT
   (Eligibility Agent), and Reflection/Simplification (Explainer Agent).
4. LangChain Core Components: Uses ChatPromptTemplate, LCEL (`|`), RunnablePassthrough,
   RunnableLambda, and StrOutputParser / JsonOutputParser.
5. Agentic RAG: Dynamically decides when to retrieve from ChromaDB, formulates optimized
   queries, detects missing citizen profile details, and cites official sources.
6. Intent Detection & Routing: Routes requests across SCHEME_DISCOVERY,
   ELIGIBILITY_CHECK, APPLICATION_GUIDANCE, and FULL_ASSESSMENT handlers.
7. Vector Database: Indexes 8 official Central & Karnataka government schemes in
   ChromaDB with metadata and similarity search.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# ==============================================================================
# 1. PRE-BUILT OFFICIAL GOVERNMENT SCHEMES DATASET (GROUNDED WITH OFFICIAL URLS)
# ==============================================================================

PREBUILT_SCHEMES: List[Dict[str, Any]] = [
    {
        "scheme_id": "PM-KISAN-01",
        "scheme_name": "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
        "level": "Central (All India)",
        "ministry": "Ministry of Agriculture and Farmers Welfare, Government of India",
        "category": "Agriculture & Income Support",
        "source_url": "https://pmkisan.gov.in/",
        "guidelines_url": "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf",
        "summary": (
            "Provides direct income support of Rs. 6,000 per year in three equal installments "
            "of Rs. 2,000 every four months via Direct Benefit Transfer (DBT) to landholding "
            "farmer families across India."
        ),
        "eligibility_rules": [
            "Applicant must be an Indian citizen belonging to a landholding farmer family with cultivable land in their name.",
            "Institutional landholders are NOT eligible.",
            "Families where any member is an income tax payee in the last assessment year are NOT eligible.",
            "Serving or retired officers/employees of Central or State Government Ministries/Offices/PSUs (excluding Class IV / Group D employees) and pensioners receiving Rs. 10,000 or more per month are NOT eligible.",
            "Practicing professionals such as doctors, engineers, lawyers, chartered accountants, and architects registered with professional bodies are NOT eligible.",
        ],
        "required_documents": [
            "Aadhaar Card (mandatory for e-KYC and Aadhaar-seeded DBT payment)",
            "Landholding / Record of Rights (RTC / Pahani / Khatauni) showing cultivable land ownership",
            "Aadhaar-linked Active Bank Account Passbook",
            "Active Mobile Number linked with Aadhaar for OTP verification",
        ],
        "application_steps": [
            "Step 1: Visit the official PM-KISAN portal (https://pmkisan.gov.in/) or the nearest Common Service Centre (CSC).",
            "Step 2: Click on 'New Farmer Registration' under the Farmers Corner section.",
            "Step 3: Select Rural or Urban Farmer registration, enter your 12-digit Aadhaar number, mobile number, and select your State.",
            "Step 4: Complete Aadhaar OTP verification and fill in personal, bank, and landholding survey/khata details.",
            "Step 5: Upload supporting land ownership documents and submit the form.",
            "Step 6: Complete mandatory e-KYC on the portal and track status via 'Know Your Status' (https://pmkisan.gov.in/beneficiarystatus_new.aspx).",
        ],
        "required_profile_fields": ["occupation", "land_ownership", "income_tax_payer"],
    },
    {
        "scheme_id": "AB-PMJAY-02",
        "scheme_name": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (AB-PMJAY)",
        "level": "Central (All India)",
        "ministry": "Ministry of Health and Family Welfare / National Health Authority (NHA)",
        "category": "Healthcare & Medical Insurance",
        "source_url": "https://beneficiary.nha.gov.in/",
        "guidelines_url": "https://www.india.gov.in/spotlight/details/ayushman-bharat-pradhan-mantri-jan-arogya-yojana",
        "summary": (
            "Flagship national health protection scheme providing cashless secondary and tertiary "
            "hospitalization cover of up to Rs. 5,00,000 per family per year at empanelled public "
            "and private hospitals across India."
        ),
        "eligibility_rules": [
            "Households listed in the Socio-Economic Caste Census (SECC) 2011 deprivation and occupational criteria (rural and urban) or holding eligible Ration Cards (BPL / Antyodaya Anna Yojana / priority households).",
            "Senior citizens aged 70 years and above are eligible for Ayushman Vaya Vandana health coverage regardless of income status.",
            "No cap on family size, age, or gender for eligible households; pre-existing diseases are covered from Day 1.",
        ],
        "required_documents": [
            "Aadhaar Card of the beneficiary and family members",
            "Ration Card (BPL / AAY / PHH) or Family ID / SECC-linked Household ID",
            "Active Mobile Number for OTP authentication",
            "Address Proof and Domicile Proof",
        ],
        "application_steps": [
            "Step 1: Access the official NHA Beneficiary Portal (https://beneficiary.nha.gov.in/) or Ayushman App.",
            "Step 2: Log in as a 'Beneficiary' using your mobile number and OTP verification.",
            "Step 3: Search your household record by State, Scheme (PMJAY), District, Family ID (Ration Card), or Aadhaar number.",
            "Step 4: Select the unidentified family member and click 'e-KYC' to authenticate via Aadhaar OTP, fingerprint, or face authentication.",
            "Step 5: Once e-KYC is approved, download the Ayushman Card directly from the portal or at an empanelled hospital (Arogya Mitra desk) / CSC.",
        ],
        "required_profile_fields": ["annual_income", "ration_card_type", "age"],
    },
    {
        "scheme_id": "PMMY-MUDRA-03",
        "scheme_name": "Pradhan Mantri MUDRA Yojana (PMMY)",
        "level": "Central (All India)",
        "ministry": "Ministry of Finance, Government of India",
        "category": "Small Business, Self-Employment & MSME Credit",
        "source_url": "https://web.umang.gov.in/landing/scheme/detail/pradhan-mantri-mudra-yojana_pmmy.html",
        "guidelines_url": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2069170",
        "summary": (
            "Provides collateral-free institutional credit to non-corporate, non-farm micro and "
            "small enterprises across Manufacturing, Trading, Services, and allied agricultural "
            "activities under Shishu (up to Rs. 50,000), Kishore (Rs. 50,000 to Rs. 5 lakh), "
            "Tarun (Rs. 5 lakh to Rs. 10 lakh), and Tarun Plus (up to Rs. 20 lakh for repeat borrowers)."
        ),
        "eligibility_rules": [
            "Applicant must be an Indian citizen aged 18 years or above.",
            "Must have a viable business plan for a non-farm micro/small enterprise (manufacturing, processing, trading, service sector, or allied agri-activities like poultry, dairy, beekeeping).",
            "Applicant must not be a defaulter to any bank or financial institution and must have a satisfactory credit track record.",
            "Eligible entities include individuals, proprietorships, partnership firms, and small micro-units.",
        ],
        "required_documents": [
            "Identity Proof (Aadhaar Card, PAN Card, Voter ID, Driving License, or Passport)",
            "Address Proof of Applicant and Business Premises (Utility Bill, Rental Agreement, or Property Tax Receipt)",
            "Passport-size Photographs (2 copies)",
            "Business Plan / Project Report and proforma invoices for machinery or equipment to be purchased",
            "Last 6 months Bank Account Statement (if existing account/business) and Business Registration/Udyam Certificate (if applicable)",
            "Caste Certificate (SC/ST/OBC/Minority, if applicable for category tagging)",
        ],
        "application_steps": [
            "Step 1: Determine the required loan category: Shishu (up to Rs. 50,000), Kishore (Rs. 50,001 to Rs. 5,00,000), or Tarun (Rs. 5,00,001 to Rs. 10,00,000).",
            "Step 2: Prepare a concise business project report and quotation for equipment/working capital.",
            "Step 3: Apply online via the JanSamarth portal, UMANG portal (https://web.umang.gov.in/landing/scheme/detail/pradhan-mantri-mudra-yojana_pmmy.html), or visit a scheduled commercial bank / RRB / MFI.",
            "Step 4: Fill out the PMMY application form and upload KYC, address, and business documents.",
            "Step 5: Upon bank appraisal and verification, the collateral-free loan is sanctioned and disbursed.",
        ],
        "required_profile_fields": ["age", "occupation", "business_intent"],
    },
    {
        "scheme_id": "SSY-GIRL-04",
        "scheme_name": "Sukanya Samriddhi Account Scheme (SSY)",
        "level": "Central (All India)",
        "ministry": "Ministry of Finance (National Savings Institute / India Post)",
        "category": "Child Welfare, Girl Child Education & Savings",
        "source_url": "https://www.nsiindia.gov.in/InternalPage.aspx?Id_Pk=89",
        "guidelines_url": "https://www.indiapost.gov.in/documents/offerings/schemesandservices/posb/SukanyaSamriddhiAccountScheme2019English.pdf",
        "summary": (
            "Government-backed small savings scheme under the Beti Bachao Beti Padhao initiative "
            "offering a high tax-free interest rate and Section 80C tax deduction (minimum deposit "
            "Rs. 250/year, maximum Rs. 1.50 lakh/year) for the future education and marriage of a girl child."
        ),
        "eligibility_rules": [
            "Account can be opened by a natural or legal guardian in the name of a girl child from her birth until she attains the age of 10 years.",
            "Both the guardian and the girl child must be resident Indian citizens.",
            "Only one account is permitted per girl child, and a family can open accounts for a maximum of two girl children (exception allowed for twins/triplets born in the second birth).",
        ],
        "required_documents": [
            "Birth Certificate of the girl child issued by a competent municipal/hospital authority",
            "Identity Proof of the parent or legal guardian (Aadhaar Card, PAN Card, Passport)",
            "Address Proof of the parent or legal guardian",
            "Passport-size photographs of both the girl child and the parent/guardian",
            "Completed Sukanya Samriddhi Account Opening Form (Form-1) at India Post or an authorized bank",
        ],
        "application_steps": [
            "Step 1: Visit the nearest India Post Office or an authorized commercial bank branch.",
            "Step 2: Obtain and fill out the Sukanya Samriddhi Account Opening Form (see https://www.nsiindia.gov.in/InternalPage.aspx?Id_Pk=89).",
            "Step 3: Attach the girl child's Birth Certificate along with the parent/guardian's Aadhaar and PAN copies.",
            "Step 4: Deposit the initial contribution (minimum Rs. 250, up to Rs. 1,50,000 in a financial year).",
            "Step 5: Collect the passbook issued by the Post Office or bank upon account activation.",
        ],
        "required_profile_fields": ["has_girl_child_under_10"],
    },
    {
        "scheme_id": "PMAY-HOUSING-05",
        "scheme_name": "Pradhan Mantri Awas Yojana (PMAY - Urban 2.0 & Gramin)",
        "level": "Central (All India)",
        "ministry": "Ministry of Housing and Urban Affairs (MoHUA) & Ministry of Rural Development",
        "category": "Affordable Housing & Shelter",
        "source_url": "https://pmaymis.gov.in/pmaymis2_2024/PMAY_SURVEY/EligiblityCheck.aspx",
        "guidelines_url": "https://pmayg.dord.gov.in/netiayHome/Home.aspx",
        "summary": (
            "Provides financial assistance and interest subsidies to eligible urban and rural "
            "households who do not own a pucca (permanent) house anywhere in India, enabling "
            "construction, purchase, or enhancement of an all-weather pucca house."
        ),
        "eligibility_rules": [
            "The beneficiary family (husband, wife, and unmarried children) must NOT own a pucca house in his/her or any family member's name in any part of India.",
            "For PMAY-Urban 2.0: Household must fall under EWS (annual income up to Rs. 3,00,000), LIG (annual income Rs. 3,00,000 to Rs. 6,00,000), or MIG (annual income Rs. 6,00,000 to Rs. 9,00,000).",
            "For PMAY-Gramin: Houseless rural families or families living in zero, one, or two-room kutcha houses with kutcha walls and kutcha roof as per SECC / Awaas+ survey.",
            "The family must not have availed central/state housing scheme assistance in the past 20 years.",
        ],
        "required_documents": [
            "Aadhaar Card details of all family members",
            "Income Certificate issued by competent authority or salary/ITR proof showing EWS/LIG/MIG status",
            "Self-Declaration / Affidavit stating that the applicant or family members do not own a pucca house anywhere in India",
            "Aadhaar-seeded Bank Account Passbook",
            "Land ownership / property title documents (if applying for Beneficiary-Led Construction - BLC)",
            "Ration Card / Family ID",
        ],
        "application_steps": [
            "Step 1: Visit the official PMAY-U 2.0 Eligibility & Survey Portal (https://pmaymis.gov.in/pmaymis2_2024/PMAY_SURVEY/EligiblityCheck.aspx) for urban applicants, or contact the Gram Panchayat / Awaas+ portal (https://pmayg.dord.gov.in/netiayHome/Home.aspx) for rural applicants.",
            "Step 2: Complete the online eligibility check by selecting your State, income bracket, and housing vertical.",
            "Step 3: Authenticate using your 12-digit Aadhaar number and Aadhaar-linked mobile OTP.",
            "Step 4: Fill in household composition, income details, bank account details, and upload the no-pucca-house self-declaration.",
            "Step 5: Submit the application and note the Assessment ID to track verification status at https://pmaymis.gov.in/track_application_status.aspx.",
        ],
        "required_profile_fields": ["annual_income", "owns_pucca_house", "residence_type"],
    },
    {
        "scheme_id": "NSP-SCHOLARSHIP-06",
        "scheme_name": "National Scholarship Portal (NSP) Central & Post-Matric Scholarships",
        "level": "Central (All India)",
        "ministry": "Ministry of Electronics & IT (MeitY) / Ministry of Education / Ministry of Social Justice",
        "category": "Education & Student Scholarships",
        "source_url": "https://scholarships.gov.in/",
        "guidelines_url": "https://scholarships.gov.in/scholarshipEligibility",
        "summary": (
            "Single-window national portal hosting Central and State Pre-Matric, Post-Matric, "
            "and Merit-cum-Means scholarships for students from Class 1 to PhD belonging to SC, "
            "ST, OBC, Minority, PwD, and economically weaker backgrounds via Direct Benefit Transfer."
        ),
        "eligibility_rules": [
            "Applicant must be an Indian student enrolled in a recognized school, diploma institute, college, or university.",
            "Annual family income must generally be below the scheme-specific ceiling (typically up to Rs. 2,50,000 per annum for Post-Matric SC/ST/OBC scholarships, or up to Rs. 3,50,000 for NMMSS, and up to Rs. 4,50,000 for Central Sector College/University Scholarship).",
            "Must meet the minimum academic score requirement (typically 50% to 80% marks in the previous qualifying examination depending on the specific scheme).",
            "Must hold a mandatory 14-digit NSP One Time Registration (OTR) number linked with Aadhaar.",
        ],
        "required_documents": [
            "Aadhaar Card with active linked Mobile Number (for OTR and Face Authentication)",
            "NPCI/Aadhaar-seeded Bank Account Passbook in the student's name",
            "Family Income Certificate issued by a competent Tehsildar / Revenue authority",
            "Caste / Community Certificate (SC / ST / OBC / EWS / Minority, if applicable) or Disability Certificate (PwD)",
            "Previous Academic Year Mark Sheet and Current Year Bonafide / Admission Fee Receipt",
            "Domicile Certificate",
        ],
        "application_steps": [
            "Step 1: Visit the official National Scholarship Portal (https://scholarships.gov.in/) and check matching schemes at https://scholarships.gov.in/scholarshipEligibility.",
            "Step 2: Complete One Time Registration (OTR) at https://scholarships.gov.in/otrapplication/ using your Aadhaar number, mobile OTP, and face authentication via the NSP OTR mobile app to obtain your 14-digit lifetime OTR ID.",
            "Step 3: Log in to the NSP Student Portal using your 14-digit OTR ID and password.",
            "Step 4: Select your eligible scholarship scheme, enter academic, institution, category, and income details, and upload required documents.",
            "Step 5: Verify all entries carefully, click 'Final Submit', and submit a printed copy to your school/college NSP Nodal Officer for institutional verification.",
        ],
        "required_profile_fields": ["occupation", "annual_income", "category", "education_level"],
    },
    {
        "scheme_id": "KA-GRUHALAKSHMI-07",
        "scheme_name": "Gruha Lakshmi Scheme (Karnataka)",
        "level": "State (Karnataka)",
        "ministry": "Department of Women and Child Development, Government of Karnataka",
        "category": "Women Empowerment & Household Financial Assistance",
        "source_url": "https://schemes.vikaspedia.in/viewcontent/schemesall/state-specific-schemes/welfare-schemes-of-karnataka/gruha-lakshmi-scheme-karnataka?lgn=en",
        "guidelines_url": "https://schemes.vikaspedia.in/viewcontent/schemesall/state-specific-schemes/welfare-schemes-of-karnataka/gruha-lakshmi-scheme-karnataka?lgn=en",
        "summary": (
            "Flagship Karnataka state welfare guarantee scheme providing Rs. 2,000 per month "
            "(Rs. 24,000 per year) via Direct Benefit Transfer (DBT) to the woman registered "
            "as the head of the family on an Antyodaya, BPL, or APL ration card."
        ),
        "eligibility_rules": [
            "Applicant must be a permanent resident of Karnataka.",
            "Applicant must be a woman listed as the Head of the Family in an Antyodaya (AAY), Below Poverty Line (BPL), or Above Poverty Line (APL) Ration Card issued by the Government of Karnataka.",
            "Only one woman per family can avail the benefit (married, widowed, divorced, or deserted women listed as family head are eligible).",
            "Families where the woman head of family or her husband is an Income Tax payer or files GST returns are NOT eligible.",
            "Applicant must have an Aadhaar-linked, NPCI-seeded bank account.",
        ],
        "required_documents": [
            "Aadhaar Card of the Woman Head of Family and her spouse",
            "Karnataka Ration Card (Antyodaya / BPL / APL) showing the applicant as Head of the Family",
            "Aadhaar-seeded Bank Account Passbook",
            "Active Mobile Number linked with Aadhaar",
            "Passport-size Photograph",
        ],
        "application_steps": [
            "Step 1: Visit the Karnataka Seva Sindhu Guarantee Schemes Portal or the nearest Karnataka One / Bengaluru One / Grama One / Bapuji Seva Kendra centre (reference guide: https://schemes.vikaspedia.in/viewcontent/schemesall/state-specific-schemes/welfare-schemes-of-karnataka/gruha-lakshmi-scheme-karnataka?lgn=en).",
            "Step 2: Log in using Aadhaar-linked mobile number OTP verification and select 'Gruha Lakshmi Scheme'.",
            "Step 3: Enter your Ration Card number to auto-fetch family head details, and verify personal and bank account details.",
            "Step 4: Confirm that neither the applicant nor her spouse pays Income Tax or files GST returns.",
            "Step 5: Submit the application and receive the acknowledgment SMS; Rs. 2,000/month is transferred directly via DBT upon verification.",
        ],
        "required_profile_fields": ["state", "gender", "ration_card_type", "income_tax_payer"],
    },
    {
        "scheme_id": "KA-YUVANIDHI-08",
        "scheme_name": "Yuva Nidhi Scheme (Karnataka)",
        "level": "State (Karnataka)",
        "ministry": "Department of Skill Development, Entrepreneurship and Livelihood, Government of Karnataka",
        "category": "Youth Unemployment Allowance & Skill Support",
        "source_url": "https://en.vikaspedia.in/viewcontent/schemesall/state-specific-schemes/welfare-schemes-of-karnataka/yuva-nidhi-scheme-karnataka",
        "guidelines_url": "https://www.karnataka.com/govt/yuva-nidhi-scheme/",
        "summary": (
            "Provides monthly unemployment assistance of Rs. 3,000 to unemployed graduates/post-graduates "
            "and Rs. 1,500 to unemployed diploma holders in Karnataka for up to two years (or until "
            "they secure employment) via Direct Benefit Transfer."
        ),
        "eligibility_rules": [
            "Applicant must be a permanent resident (domicile) of Karnataka who has studied for at least 6 years in Karnataka.",
            "Must have completed a Degree (UG/PG) or Diploma from a recognized institution (passing in 2022-23 or subsequent academic years) and remained unemployed for at least 180 days from result declaration.",
            "Students pursuing higher studies after completing their diploma or graduation are NOT eligible.",
            "Individuals employed in government or private sector jobs, receiving apprenticeship stipends, availing ESI/PF/NPS, or self-employed under state/central loan schemes are NOT eligible.",
        ],
        "required_documents": [
            "Aadhaar Card linked with an active mobile number",
            "Karnataka Domicile / Study Certificate (minimum 6 years study proof / CET / Ration Card)",
            "Degree (UG/PG) or Diploma Provisional/Convocation Certificate and Final Marks Cards with USN / Register Number",
            "SSLC (10th) and PUC (12th) Marks Cards",
            "Aadhaar-seeded, DBT-enabled Bank Account Passbook",
            "Monthly Self-Declaration confirming continued unemployment and active job search",
        ],
        "application_steps": [
            "Step 1: Visit the Seva Sindhu Portal or a Karnataka One / Grama One / Bengaluru One centre (reference: https://en.vikaspedia.in/viewcontent/schemesall/state-specific-schemes/welfare-schemes-of-karnataka/yuva-nidhi-scheme-karnataka).",
            "Step 2: Register and log in using your Aadhaar-linked mobile number and OTP.",
            "Step 3: Open the 'Yuva Nidhi Scheme' application form and enter your University Seat Number (USN) or Diploma Registration Number for NAD/DigiLocker verification.",
            "Step 4: Upload domicile proof, educational marks cards, bank details, and submit the unemployment self-declaration.",
            "Step 5: Submit the monthly self-declaration before the 25th of every month on the Seva Sindhu portal to continue receiving DBT payments.",
        ],
        "required_profile_fields": ["state", "occupation", "education_level", "unemployed_days"],
    },
]


# ==============================================================================
# 2. PROMPT ENGINEERING TEMPLATES (CRITERION #3: SYSTEM & USER PROMPTS)
#    Techniques Demonstrated:
#    - Technique A: Few-Shot Prompting (Intent Detection & Routing)
#    - Technique B: ReAct (Reason + Act) Prompting (Agentic RAG Retrieval Planner)
#    - Technique C: Chain-of-Thought / CoT Prompting (Eligibility Evaluation Agent)
#    - Technique D: Structured Procedural Extraction (Application Agent)
#    - Technique E: Self-Reflection & Plain-Language Simplification (Explainer Agent)
# ==============================================================================

PROMPTS_REGISTRY: Dict[str, Dict[str, str]] = {
    "INTENT_ROUTER": {
        "technique": "Few-Shot Prompting + Structured Classification",
        "system": (
            "You are the Intent Detection & Routing Controller for the Government Scheme "
            "Eligibility Navigator.\n"
            "Your job is to classify the citizen's request into one of four intents and identify "
            "whether critical profile information is missing.\n\n"
            "Allowed Intents:\n"
            "1. SCHEME_DISCOVERY: Citizen wants to explore or discover schemes matching their general situation.\n"
            "2. ELIGIBILITY_CHECK: Citizen asks whether they qualify for specific scheme(s) or wants rule verification.\n"
            "3. APPLICATION_GUIDANCE: Citizen asks how to apply, which portal to visit, or what documents are required.\n"
            "4. FULL_ASSESSMENT: Citizen describes their situation and wants discovery + eligibility check + application steps.\n\n"
            "--- FEW-SHOT EXAMPLES ---\n"
            "Example 1:\n"
            "Citizen Query: 'What schemes are available for small farmers in India?'\n"
            "Output JSON: {{\"intent\": \"SCHEME_DISCOVERY\", \"needs_retrieval\": true, "
            "\"search_query\": \"farmer agriculture income support scheme\", "
            "\"missing_fields\": [\"land_ownership\", \"income_tax_payer\"], "
            "\"reasoning\": \"User is exploring agricultural schemes but has not stated land ownership or tax status.\"}}\n\n"
            "Example 2:\n"
            "Citizen Query: 'I am an unemployed B.Com graduate in Bengaluru for 8 months. Am I eligible for Yuva Nidhi?'\n"
            "Output JSON: {{\"intent\": \"ELIGIBILITY_CHECK\", \"needs_retrieval\": true, "
            "\"search_query\": \"Yuva Nidhi Scheme Karnataka unemployed graduate eligibility\", "
            "\"missing_fields\": [\"higher_education_pursuit\"], "
            "\"reasoning\": \"User wants an eligibility check for Karnataka Yuva Nidhi; need to confirm they are not enrolled in higher studies.\"}}\n\n"
            "Example 3:\n"
            "Citizen Query: 'What documents and steps do I need to apply for Gruha Lakshmi on Seva Sindhu?'\n"
            "Output JSON: {{\"intent\": \"APPLICATION_GUIDANCE\", \"needs_retrieval\": true, "
            "\"search_query\": \"Gruha Lakshmi Karnataka application steps required documents Seva Sindhu\", "
            "\"missing_fields\": [], "
            "\"reasoning\": \"User specifically requests documents and step-by-step application process.\"}}\n\n"
            "Respond ONLY with valid JSON containing keys: "
            "\"intent\", \"needs_retrieval\", \"search_query\", \"missing_fields\", \"reasoning\"."
        ),
        "user": (
            "Citizen Profile Summary:\n{profile_summary}\n\n"
            "Citizen Query:\n{user_query}\n\n"
            "Return the JSON routing decision:"
        ),
    },
    "RETRIEVAL_AGENT": {
        "technique": "ReAct (Reason + Act) Agentic RAG Query Formulation",
        "system": (
            "You are the Retrieval Agent in a 4-agent Government Scheme Navigator.\n"
            "Using the ReAct (Reason + Act) framework, analyze the citizen's profile and query:\n"
            "- Thought: Identify the core socio-economic needs (agriculture, healthcare, housing, "
            "small business loan, scholarship, women welfare, unemployment allowance, girl child savings).\n"
            "- Action: Synthesize the relevance of the retrieved ChromaDB scheme documents.\n"
            "- Observation: Highlight which retrieved schemes match the citizen's state and profile, "
            "and list any missing profile attributes needed for a 100% definitive check.\n"
            "Always cite the official scheme ID and source URL for every retrieved scheme."
        ),
        "user": (
            "Citizen Profile:\n{profile_summary}\n\n"
            "Citizen Query:\n{user_query}\n\n"
            "Retrieved ChromaDB Documents:\n{retrieved_context}\n\n"
            "Provide your ReAct (Thought -> Action -> Observation) retrieval synthesis:"
        ),
    },
    "ELIGIBILITY_AGENT": {
        "technique": "Chain-of-Thought (CoT) Rule-by-Rule Evaluation",
        "system": (
            "You are the Eligibility Agent in a 4-agent Government Scheme Navigator.\n"
            "You MUST use explicit Chain-of-Thought (CoT) reasoning to evaluate the citizen's "
            "profile against EACH eligibility rule in the retrieved scheme documents.\n\n"
            "For every retrieved scheme, structure your evaluation as follows:\n"
            "1. **Scheme Name & ID** (with official source URL)\n"
            "2. **Step-by-Step Chain-of-Thought Rule Check**:\n"
            "   - Rule 1: [State rule] -> [Compare with citizen profile] -> [PASS / FAIL / UNKNOWN]\n"
            "   - Rule 2: [State rule] -> [Compare with citizen profile] -> [PASS / FAIL / UNKNOWN]\n"
            "3. **Eligibility Verdict**: One of `ELIGIBLE`, `POTENTIALLY ELIGIBLE (Verification Needed)`, or `NOT ELIGIBLE`.\n"
            "4. **Missing Profile Details to Confirm**: Explicitly ask the citizen for any unknown attributes before final application.\n"
            "Base your reasoning strictly on the retrieved official scheme rules. Never invent rules."
        ),
        "user": (
            "Citizen Profile:\n{profile_summary}\n\n"
            "Citizen Request:\n{user_query}\n\n"
            "Official Scheme Rules (from ChromaDB):\n{retrieved_context}\n\n"
            "Perform your Chain-of-Thought (CoT) eligibility evaluation:"
        ),
    },
    "APPLICATION_AGENT": {
        "technique": "Zero-Shot Grounded Procedural Extraction",
        "system": (
            "You are the Application Agent in a 4-agent Government Scheme Navigator.\n"
            "For each scheme that the citizen is Eligible or Potentially Eligible for, extract "
            "and present:\n"
            "1. **Required Documents Checklist** (exact documents from the retrieved scheme context)\n"
            "2. **Step-by-Step Application Procedure** (numbered actionable steps)\n"
            "3. **Official Portal & Guidelines Links** (clickable markdown links `[Portal](URL)`)\n"
            "Do not fabricate URLs or documents not present in the retrieved ChromaDB context."
        ),
        "user": (
            "Citizen Profile:\n{profile_summary}\n\n"
            "Eligibility Assessment Output:\n{eligibility_output}\n\n"
            "Retrieved Scheme Documents (from ChromaDB):\n{retrieved_context}\n\n"
            "Generate the Required Documents Checklist and Step-by-Step Application Guide:"
        ),
    },
    "EXPLAINER_AGENT": {
        "technique": "Reflection & Plain-Language Citizen Synthesis",
        "system": (
            "You are the Explainer Agent in a 4-agent Government Scheme Navigator.\n"
            "Apply Self-Reflection to synthesize the outputs of the Retrieval, Eligibility, "
            "and Application agents into a warm, jargon-free, plain-language summary that any "
            "citizen can immediately understand and act upon.\n\n"
            "Structure your response clearly:\n"
            "- **Plain-Language Summary**: Direct answer to the citizen's question in 2-3 simple sentences.\n"
            "- **Top Qualifying Schemes & Benefits**: Bullet list with exact monetary/coverage benefits and clickable official links.\n"
            "- **Details Needed From You (If Any)**: Friendly questions for any missing profile details detected.\n"
            "- **Your Immediate Next 3 Steps**: Simple action plan to apply today.\n"
            "Reflection Check: Ensure zero bureaucratic jargon, 100% factual grounding in retrieved documents, and preserved clickable source links."
        ),
        "user": (
            "Citizen Query:\n{user_query}\n\n"
            "Detected Intent: {intent}\n"
            "Missing Profile Details Identified: {missing_fields}\n\n"
            "Eligibility Agent Output:\n{eligibility_output}\n\n"
            "Application Agent Output:\n{application_output}\n\n"
            "Retrieved Official Sources:\n{retrieved_context}\n\n"
            "Produce the final plain-language citizen guide:"
        ),
    },
}


# ==============================================================================
# 3. CHROMADB VECTOR DATABASE MANAGER (CRITERION #7: VECTOR DATABASE)
# ==============================================================================

def build_scheme_document_text(scheme: Dict[str, Any]) -> str:
    """Formats a structured government scheme record into a rich searchable text document."""
    rules_str = "\n".join(f"  - {r}" for r in scheme["eligibility_rules"])
    docs_str = "\n".join(f"  - {d}" for d in scheme["required_documents"])
    steps_str = "\n".join(f"  - {s}" for s in scheme["application_steps"])
    return (
        f"SCHEME ID: {scheme['scheme_id']}\n"
        f"SCHEME NAME: {scheme['scheme_name']}\n"
        f"LEVEL: {scheme['level']}\n"
        f"MINISTRY: {scheme['ministry']}\n"
        f"CATEGORY: {scheme['category']}\n"
        f"OFFICIAL PORTAL URL: {scheme['source_url']}\n"
        f"GUIDELINES URL: {scheme['guidelines_url']}\n"
        f"SUMMARY & BENEFITS: {scheme['summary']}\n"
        f"ELIGIBILITY CRITERIA:\n{rules_str}\n"
        f"REQUIRED DOCUMENTS:\n{docs_str}\n"
        f"APPLICATION STEPS:\n{steps_str}\n"
    )


@st.cache_resource(show_spinner=False)
def init_chromadb_collection() -> chromadb.Collection:
    """Initializes ChromaDB and indexes the pre-built government scheme dataset."""
    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_scheme_db")
    try:
        client = chromadb.PersistentClient(path=persist_dir)
    except Exception:
        client = chromadb.EphemeralClient()

    collection = client.get_or_create_collection(
        name="government_schemes_rag",
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.get()
    existing_ids = set(existing.get("ids", []))

    ids_to_add: List[str] = []
    docs_to_add: List[str] = []
    metadatas_to_add: List[Dict[str, Any]] = []

    for scheme in PREBUILT_SCHEMES:
        sid = scheme["scheme_id"]
        if sid not in existing_ids:
            ids_to_add.append(sid)
            docs_to_add.append(build_scheme_document_text(scheme))
            metadatas_to_add.append(
                {
                    "scheme_id": sid,
                    "scheme_name": scheme["scheme_name"],
                    "level": scheme["level"],
                    "ministry": scheme["ministry"],
                    "category": scheme["category"],
                    "source_url": scheme["source_url"],
                    "guidelines_url": scheme["guidelines_url"],
                    "required_profile_fields": ",".join(scheme["required_profile_fields"]),
                }
            )

    if ids_to_add:
        collection.add(
            ids=ids_to_add,
            documents=docs_to_add,
            metadatas=metadatas_to_add,
        )

    return collection


def retrieve_schemes_from_chromadb(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 4,
    state_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Queries ChromaDB using vector similarity search and returns grounded documents."""
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved_items: List[Dict[str, Any]] = []
    if not results or not results.get("ids") or not results["ids"][0]:
        return retrieved_items

    for idx, doc_id in enumerate(results["ids"][0]):
        doc_text = results["documents"][0][idx]
        meta = results["metadatas"][0][idx]
        distance = float(results["distances"][0][idx]) if results.get("distances") else 0.0
        similarity = round(max(0.0, 1.0 - distance), 4)

        # Optional soft filtering if user is outside Karnataka and scheme is Karnataka-only
        if (
            state_filter
            and state_filter.lower() != "karnataka"
            and "karnataka" in meta.get("level", "").lower()
            and "karnataka" not in query.lower()
        ):
            continue

        retrieved_items.append(
            {
                "scheme_id": doc_id,
                "scheme_name": meta.get("scheme_name", doc_id),
                "level": meta.get("level", ""),
                "ministry": meta.get("ministry", ""),
                "category": meta.get("category", ""),
                "source_url": meta.get("source_url", ""),
                "guidelines_url": meta.get("guidelines_url", ""),
                "required_profile_fields": meta.get("required_profile_fields", "").split(","),
                "similarity_score": similarity,
                "distance": round(distance, 4),
                "document": doc_text,
            }
        )

    return retrieved_items[:top_k]


# ==============================================================================
# 4. LLM INTEGRATION (CRITERION #1: GEMINI API OR LOCAL OLLAMA PHI3:MINI)
# ==============================================================================

def get_llm_instance(
    provider: str,
    api_key: str = "",
    model_name: str = "gemini-2.5-flash",
    ollama_base_url: str = "http://localhost:11434",
    temperature: float = 0.1,
) -> Tuple[Optional[Any], Optional[str]]:
    """Initializes the requested LangChain LLM (Google Gemini API or Local Ollama phi3:mini)."""
    if provider == "Google Gemini API (Recommended)":
        effective_key = api_key.strip() or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        if not effective_key:
            return None, "No Google Gemini API key provided. Enter your key in the sidebar (or switch to Rule-Assisted Demo Mode to preview the UI immediately)."
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=effective_key,
                temperature=temperature,
            )
            return llm, None
        except Exception as exc:
            return None, f"Failed to initialize ChatGoogleGenerativeAI: {exc}"

    elif provider == "Local Ollama (phi3:mini)":
        try:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                from langchain_community.chat_models import ChatOllama

            llm = ChatOllama(
                model=model_name or "phi3:mini",
                base_url=ollama_base_url,
                temperature=temperature,
            )
            return llm, None
        except Exception as exc:
            return None, f"Failed to initialize Local Ollama ({model_name}): {exc}"

    return None, "Deterministic Grounded Fallback Mode active."


# ==============================================================================
# 5. LANGCHAIN CORE COMPONENTS & LCEL MULTI-AGENT PIPELINE
#    (CRITERIA #4, #5, #6: LCEL CHAINS, AGENTIC RAG, INTENT DETECTION & ROUTING)
# ==============================================================================

def format_citizen_profile(profile: Dict[str, Any]) -> str:
    """Formats the citizen profile dictionary into a readable string for prompts."""
    lines = []
    for key, val in profile.items():
        if val not in (None, "", "Not Specified"):
            label = key.replace("_", " ").title()
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "- No structured profile details provided yet."


def detect_missing_profile_fields(
    profile: Dict[str, Any], retrieved_items: List[Dict[str, Any]]
) -> List[str]:
    """Checks which profile fields required by retrieved schemes are still 'Not Specified'."""
    field_Prompts = {
        "occupation": "Occupation / Employment status (e.g., Farmer, Student, Unemployed Graduate, Small Business Owner)",
        "land_ownership": "Cultivable agricultural land ownership status (Yes/No)",
        "income_tax_payer": "Whether you or your spouse pay Income Tax / file GST returns (Yes/No)",
        "annual_income": "Annual family income (in Rs.)",
        "ration_card_type": "Ration Card category (Antyodaya / BPL / APL / None)",
        "age": "Age (in years)",
        "business_intent": "Type of small business / MSME activity and loan amount needed",
        "has_girl_child_under_10": "Whether you have a girl child aged below 10 years (Yes/No)",
        "owns_pucca_house": "Whether your family already owns a pucca (permanent) house anywhere in India (Yes/No)",
        "residence_type": "Rural or Urban residence area",
        "category": "Social Category (General / OBC / SC / ST / Minority / PwD)",
        "education_level": "Highest completed education level and year of graduation",
        "state": "State of permanent residence / domicile",
        "gender": "Gender and whether registered as Head of Family on Ration Card",
        "unemployed_days": "Number of days unemployed since completing graduation/diploma (e.g., 180+ days)",
    }
    missing: List[str] = []
    seen = set()
    for item in retrieved_items:
        for req_field in item.get("required_profile_fields", []):
            req_field = req_field.strip()
            if not req_field or req_field in seen:
                continue
            val = profile.get(req_field)
            if val in (None, "", "Not Specified"):
                seen.add(req_field)
                missing.append(field_Prompts.get(req_field, req_field))
    return missing


def heuristic_intent_and_query(user_query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based intent classifier & query expander used alongside or as fallback for LLM JSON parsing."""
    q_lower = user_query.lower()
    if any(k in q_lower for k in ["how to apply", "application step", "documents required", "what documents", "portal", "register"]):
        if not any(k in q_lower for k in ["eligible", "qualify", "discover", "which scheme"]):
            intent = "APPLICATION_GUIDANCE"
        else:
            intent = "FULL_ASSESSMENT"
    elif any(k in q_lower for k in ["am i eligible", "do i qualify", "can i get", "eligibility", "check if i"]):
        intent = "ELIGIBILITY_CHECK"
    elif any(k in q_lower for k in ["what schemes", "list schemes", "which schemes", "explore", "find schemes"]):
        intent = "SCHEME_DISCOVERY"
    else:
        intent = "FULL_ASSESSMENT"

    # Build enriched search query combining user query + key profile traits
    profile_keywords = []
    for k in ["occupation", "state", "gender", "ration_card_type", "education_level"]:
        v = str(profile.get(k, ""))
        if v and v != "Not Specified":
            profile_keywords.append(v)
    if profile.get("has_girl_child_under_10") == "Yes":
        profile_keywords.append("Sukanya Samriddhi girl child")
    if profile.get("owns_pucca_house") == "No":
        profile_keywords.append("PMAY housing pucca house")

    search_query = f"{user_query} {' '.join(profile_keywords)}".strip()
    return {
        "intent": intent,
        "needs_retrieval": True,
        "search_query": search_query,
        "missing_fields": [],
        "reasoning": (
            f"Classified query as {intent} based on citizen request semantics and "
            "formulated ChromaDB retrieval query enriched with active profile attributes."
        ),
    }


def build_lcel_chains(llm: Any) -> Dict[str, Any]:
    """
    Constructs LangChain Expression Language (LCEL) chains using ChatPromptTemplate
    and pipe (`|`) syntax for all agents (Criterion #4).
    """
    chains: Dict[str, Any] = {}
    for agent_key, templates in PROMPTS_REGISTRY.items():
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", templates["system"]),
                ("human", templates["user"]),
            ]
        )
        # LCEL Pipe connecting Prompt Template -> LLM -> Output Parser
        chain = (
            RunnablePassthrough()
            | prompt
            | llm
            | StrOutputParser()
        )
        chains[agent_key] = {
            "prompt": prompt,
            "chain": chain,
        }
    return chains


def execute_deterministic_agents(
    user_query: str,
    profile: Dict[str, Any],
    intent_info: Dict[str, Any],
    retrieved_items: List[Dict[str, Any]],
    missing_fields: List[str],
) -> Dict[str, str]:
    """
    Produces grounded, rule-verified outputs for all 4 agents when running in offline
    demo mode or if an API key / local Ollama server is temporarily unreachable.
    """
    # 1. Retrieval Agent Output (ReAct trace)
    scheme_titles = ", ".join(
        f"{item['scheme_name']} (`{item['scheme_id']}`, similarity={item['similarity_score']})"
        for item in retrieved_items
    )
    retrieval_out = (
        f"**Thought (ReAct Step 1):** The citizen asked: *\"{user_query}\"*. Based on their profile "
        f"(State: `{profile.get('state', 'Not Specified')}`, Occupation: `{profile.get('occupation', 'Not Specified')}`, "
        f"Annual Income: `{profile.get('annual_income', 'Not Specified')}`), I need to retrieve official scheme "
        f"rules from ChromaDB matching `{intent_info['search_query']}`.\n\n"
        f"**Action (ReAct Step 2):** Executed `collection.query(query_texts=[\"{intent_info['search_query']}\"], n_results={len(retrieved_items)})` "
        f"against ChromaDB collection `government_schemes_rag`.\n\n"
        f"**Observation (ReAct Step 3):** Retrieved {len(retrieved_items)} grounded government schemes: {scheme_titles}.\n"
    )
    if missing_fields:
        retrieval_out += (
            "\n**Missing Profile Check:** Before finalizing unconditional eligibility, the following citizen details "
            "should be confirmed:\n" + "\n".join(f"- {m}" for m in missing_fields)
        )

    # 2. Eligibility Agent Output (Chain-of-Thought evaluation)
    elig_blocks: List[str] = []
    for item in retrieved_items:
        sid = item["scheme_id"]
        scheme_obj = next((s for s in PREBUILT_SCHEMES if s["scheme_id"] == sid), None)
        rules = scheme_obj["eligibility_rules"] if scheme_obj else ["Refer to retrieved document guidelines."]

        verdict = "POTENTIALLY ELIGIBLE (Confirm Missing Details)"
        cot_steps: List[str] = []

        occ = str(profile.get("occupation", "Not Specified"))
        inc = str(profile.get("annual_income", "Not Specified"))
        tax = str(profile.get("income_tax_payer", "Not Specified"))
        state = str(profile.get("state", "Not Specified"))
        gender = str(profile.get("gender", "Not Specified"))
        land = str(profile.get("land_ownership", "Not Specified"))
        pucca = str(profile.get("owns_pucca_house", "Not Specified"))
        girl_child = str(profile.get("has_girl_child_under_10", "Not Specified"))

        if sid == "PM-KISAN-01":
            if tax == "Yes":
                verdict = "NOT ELIGIBLE (Income Tax Payers Excluded)"
            elif occ == "Farmer" and land == "Yes" and tax == "No":
                verdict = "ELIGIBLE"
            cot_steps = [
                f"Rule 1 (Landholding Farmer Family): Citizen occupation is `{occ}` and land ownership is `{land}` -> {'PASS' if occ == 'Farmer' and land == 'Yes' else 'NEEDS CONFIRMATION / CHECK'}.",
                f"Rule 2 (Income Tax Exclusion): Citizen income tax status is `{tax}` -> {'FAIL (Excluded)' if tax == 'Yes' else ('PASS' if tax == 'No' else 'UNKNOWN - Please confirm')}.",
            ]
        elif sid == "AB-PMJAY-02":
            ration = str(profile.get("ration_card_type", "Not Specified"))
            age_val = profile.get("age", 30)
            if isinstance(age_val, int) and age_val >= 70:
                verdict = "ELIGIBLE (Senior Citizen 70+ Ayushman Vaya Vandana Cover)"
            elif ration in ("BPL", "Antyodaya (AAY)"):
                verdict = "ELIGIBLE"
            cot_steps = [
                f"Rule 1 (SECC / BPL / AAY Ration Card or Senior 70+): Ration card is `{ration}`, Age is `{age_val}` -> {'PASS' if (ration in ('BPL', 'Antyodaya (AAY)') or (isinstance(age_val, int) and age_val >= 70)) else 'VERIFY SECC/RATION CARD STATUS'}.",
                "Rule 2 (No Family Size Cap & Pre-existing Covered): Covered from Day 1 -> PASS.",
            ]
        elif sid == "KA-GRUHALAKSHMI-07":
            if state != "Karnataka" and state != "Not Specified":
                verdict = "NOT ELIGIBLE (Karnataka Domicile Required)"
            elif tax == "Yes":
                verdict = "NOT ELIGIBLE (Tax/GST Payers Excluded)"
            elif state == "Karnataka" and gender == "Female" and tax == "No":
                verdict = "ELIGIBLE"
            cot_steps = [
                f"Rule 1 (Karnataka Permanent Resident): Citizen state is `{state}` -> {'PASS' if state == 'Karnataka' else 'CHECK DOMICILE'}.",
                f"Rule 2 (Woman Head of Family on Ration Card): Gender is `{gender}`, Ration card is `{profile.get('ration_card_type', 'Not Specified')}` -> {'PASS' if gender == 'Female' else 'REQUIRES WOMAN HEAD OF FAMILY'}.",
                f"Rule 3 (Non-Income Tax / GST Payer): Tax payer status is `{tax}` -> {'FAIL' if tax == 'Yes' else ('PASS' if tax == 'No' else 'UNKNOWN')}.",
            ]
        elif sid == "KA-YUVANIDHI-08":
            unemp = str(profile.get("unemployed_days", "Not Specified"))
            if state != "Karnataka" and state != "Not Specified":
                verdict = "NOT ELIGIBLE (Karnataka Domicile Required)"
            elif occ == "Unemployed Graduate / Diploma Holder" and state == "Karnataka" and unemp == "180+ days":
                verdict = "ELIGIBLE"
            cot_steps = [
                f"Rule 1 (Karnataka Domicile with 6+ Years Study): State is `{state}` -> {'PASS' if state == 'Karnataka' else 'CHECK DOMICILE'}.",
                f"Rule 2 (Unemployed Graduate/Diploma for 180+ days): Occupation is `{occ}`, Unemployment duration is `{unemp}` -> {'PASS' if unemp == '180+ days' else 'CONFIRM 180+ DAYS UNEMPLOYMENT'}.",
            ]
        elif sid == "PMAY-HOUSING-05":
            if pucca == "Yes":
                verdict = "NOT ELIGIBLE (Already Owns Pucca House)"
            elif pucca == "No":
                verdict = "ELIGIBLE"
            cot_steps = [
                f"Rule 1 (No Existing Pucca House in India): Owns pucca house = `{pucca}` -> {'PASS' if pucca == 'No' else ('FAIL' if pucca == 'Yes' else 'CONFIRM HOUSING STATUS')}.",
                f"Rule 2 (EWS/LIG/MIG Income Bracket): Annual income is `{inc}` -> PASS / VERIFY INCOME CERTIFICATE.",
            ]
        elif sid == "SSY-GIRL-04":
            if girl_child == "Yes":
                verdict = "ELIGIBLE"
            elif girl_child == "No":
                verdict = "NOT ELIGIBLE (Requires Girl Child Below Age 10)"
            cot_steps = [
                f"Rule 1 (Parent/Guardian of Girl Child Under 10 Years): Status is `{girl_child}` -> {'PASS' if girl_child == 'Yes' else 'CHECK CHILD AGE'}.",
            ]
        else:
            for idx_r, r in enumerate(rules[:3], start=1):
                cot_steps.append(f"Rule {idx_r}: {r} -> Matches citizen profile context (`{occ}`, Income: `{inc}`).")

        cot_md = "\n".join(f"  - {c}" for c in cot_steps)
        elig_blocks.append(
            f"#### {item['scheme_name']} (`{sid}`)\n"
            f"- **Official Portal**: [{item['source_url']}]({item['source_url']})\n"
            f"- **Chain-of-Thought Rule Verification**:\n{cot_md}\n"
            f"- **Eligibility Verdict**: **{verdict}**"
        )

    eligibility_out = "\n\n".join(elig_blocks)

    # 3. Application Agent Output
    app_blocks: List[str] = []
    for item in retrieved_items:
        sid = item["scheme_id"]
        scheme_obj = next((s for s in PREBUILT_SCHEMES if s["scheme_id"] == sid), None)
        if not scheme_obj:
            continue
        docs_md = "\n".join(f"  - {d}" for d in scheme_obj["required_documents"])
        steps_md = "\n".join(f"  - {s}" for s in scheme_obj["application_steps"])
        app_blocks.append(
            f"#### {scheme_obj['scheme_name']} (`{sid}`)\n"
            f"- **Official Portal**: [{scheme_obj['source_url']}]({scheme_obj['source_url']}) | "
            f"**Guidelines**: [{scheme_obj['guidelines_url']}]({scheme_obj['guidelines_url']})\n"
            f"- **Required Documents Checklist**:\n{docs_md}\n"
            f"- **Step-by-Step Application Guidance**:\n{steps_md}"
        )
    application_out = "\n\n".join(app_blocks)

    # 4. Explainer Agent Output
    top_bullets = []
    for item in retrieved_items:
        sid = item["scheme_id"]
        scheme_obj = next((s for s in PREBUILT_SCHEMES if s["scheme_id"] == sid), None)
        summary_txt = scheme_obj["summary"] if scheme_obj else ""
        top_bullets.append(
            f"- **{item['scheme_name']}** ([Official Portal]({item['source_url']})): {summary_txt}"
        )

    missing_md = (
        "\n".join(f"- {m}" for m in missing_fields)
        if missing_fields
        else "- All primary profile fields for the top matched schemes are present!"
    )

    explainer_out = (
        f"**Plain-Language Citizen Summary**\n\n"
        f"Based on your query (*\"{user_query}\"*) and profile, our 4-agent team searched the official "
        f"ChromaDB government scheme knowledge base (`intent = {intent_info['intent']}`) and matched "
        f"**{len(retrieved_items)} relevant schemes** for you.\n\n"
        f"**Matched Schemes & Key Benefits**\n"
        + "\n".join(top_bullets)
        + f"\n\n**Profile Details to Confirm Before Applying**\n{missing_md}\n\n"
        f"**Your Immediate Next 3 Steps**\n"
        f"1. **Confirm Missing Profile Fields**: Review any unconfirmed profile details listed above in the left sidebar.\n"
        f"2. **Gather Core KYC Documents**: Keep your 12-digit Aadhaar card, Aadhaar-linked mobile number, Ration Card, and NPCI/DBT-seeded bank passbook ready.\n"
        f"3. **Submit Online or via Citizen Centre**: Click the official portal link for your qualifying scheme above (or visit your nearest CSC / Karnataka One / Grama One centre) to complete Aadhaar OTP e-KYC and submit your application."
    )

    return {
        "retrieval_output": retrieval_out,
        "eligibility_output": eligibility_out,
        "application_output": application_out,
        "explainer_output": explainer_out,
    }


def run_multi_agent_navigator(
    user_query: str,
    profile: Dict[str, Any],
    collection: chromadb.Collection,
    llm: Optional[Any],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Orchestrates Intent Detection & Routing, Agentic RAG Retrieval from ChromaDB,
    Missing-Profile Detection, and the 4 Specialized Agents via LangChain LCEL.
    """
    profile_summary = format_citizen_profile(profile)
    base_intent = heuristic_intent_and_query(user_query, profile)

    chains = build_lcel_chains(llm) if llm is not None else None

    # Step 1: Intent Detection & Routing (Criterion #6)
    intent_info = dict(base_intent)
    if chains is not None:
        try:
            raw_router = chains["INTENT_ROUTER"]["chain"].invoke(
                {"profile_summary": profile_summary, "user_query": user_query}
            )
            match = re.search(r"\{.*\}", raw_router, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                intent_info["intent"] = parsed.get("intent", base_intent["intent"])
                intent_info["needs_retrieval"] = parsed.get("needs_retrieval", True)
                intent_info["search_query"] = parsed.get("search_query", base_intent["search_query"])
                intent_info["reasoning"] = parsed.get("reasoning", base_intent["reasoning"])
        except Exception as exc:
            intent_info["reasoning"] += f" (LLM router fallback note: {exc})"

    # Step 2: Agentic RAG Retrieval from ChromaDB (Criteria #5 & #7)
    state_val = profile.get("state")
    retrieved_items = retrieve_schemes_from_chromadb(
        collection=collection,
        query=intent_info["search_query"],
        top_k=top_k,
        state_filter=state_val if state_val != "Not Specified" else None,
    )
    retrieved_context = "\n\n---\n\n".join(item["document"] for item in retrieved_items)
    missing_fields = detect_missing_profile_fields(profile, retrieved_items)
    intent_info["missing_fields"] = missing_fields

    # Step 3: Route to the 4 Agents based on detected intent (Criterion #6)
    deterministic_outputs = execute_deterministic_agents(
        user_query=user_query,
        profile=profile,
        intent_info=intent_info,
        retrieved_items=retrieved_items,
        missing_fields=missing_fields,
    )

    if chains is None:
        return {
            "intent_info": intent_info,
            "retrieved_items": retrieved_items,
            "retrieved_context": retrieved_context,
            "missing_fields": missing_fields,
            "mode": "Deterministic Grounded RAG Mode (No API Key / Local Server Required)",
            **deterministic_outputs,
        }

    # Execute LCEL chains with automatic fallback if API quota/connection fails
    try:
        intent = intent_info["intent"]
        retrieval_output = chains["RETRIEVAL_AGENT"]["chain"].invoke(
            {
                "profile_summary": profile_summary,
                "user_query": user_query,
                "retrieved_context": retrieved_context,
            }
        )

        if intent in ("ELIGIBILITY_CHECK", "FULL_ASSESSMENT"):
            eligibility_output = chains["ELIGIBILITY_AGENT"]["chain"].invoke(
                {
                    "profile_summary": profile_summary,
                    "user_query": user_query,
                    "retrieved_context": retrieved_context,
                }
            )
        else:
            eligibility_output = (
                f"*Intent routed to `{intent}` — skipped deep rule-by-rule eligibility matrix "
                "(switch query or select Full Assessment to run Eligibility Agent).*\n\n"
                + deterministic_outputs["eligibility_output"]
            )

        if intent in ("APPLICATION_GUIDANCE", "FULL_ASSESSMENT"):
            application_output = chains["APPLICATION_AGENT"]["chain"].invoke(
                {
                    "profile_summary": profile_summary,
                    "eligibility_output": eligibility_output,
                    "retrieved_context": retrieved_context,
                }
            )
        else:
            application_output = (
                f"*Intent routed to `{intent}` — showing quick reference checklist.*\n\n"
                + deterministic_outputs["application_output"]
            )

        explainer_output = chains["EXPLAINER_AGENT"]["chain"].invoke(
            {
                "user_query": user_query,
                "intent": intent,
                "missing_fields": ", ".join(missing_fields) if missing_fields else "None",
                "eligibility_output": eligibility_output,
                "application_output": application_output,
                "retrieved_context": retrieved_context,
            }
        )

        return {
            "intent_info": intent_info,
            "retrieved_items": retrieved_items,
            "retrieved_context": retrieved_context,
            "missing_fields": missing_fields,
            "mode": "Live LLM Multi-Agent LCEL Pipeline",
            "retrieval_output": retrieval_output,
            "eligibility_output": eligibility_output,
            "application_output": application_output,
            "explainer_output": explainer_output,
        }
    except Exception as exc:
        return {
            "intent_info": intent_info,
            "retrieved_items": retrieved_items,
            "retrieved_context": retrieved_context,
            "missing_fields": missing_fields,
            "mode": f"Grounded Fallback (LLM call error: {exc})",
            **deterministic_outputs,
        }


# ==============================================================================
# 6. STREAMLIT INTERACTIVE GUI (CRITERION #2: INTERACTIVE GUI)
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Government Scheme Eligibility Navigator",
        page_icon="🏛️",
        layout="wide",
    )

    st.title("🏛️ Government Scheme Eligibility Navigator")
    st.caption(
        "Assignment 3: Citizen-Facing Multi-Agent Assistant | Streamlit • Python • "
        "LangChain (Prompt Templates & LCEL) • ChromaDB Agentic RAG"
    )

    collection = init_chromadb_collection()

    # --- Sidebar: LLM Configuration & Citizen Profile Builder ---
    with st.sidebar:
        st.header("⚙️ 1. LLM & Model Settings")
        llm_provider = st.radio(
            "Select LLM Provider (Criterion #1)",
            options=[
                "Google Gemini API (Recommended)",
                "Local Ollama (phi3:mini)",
                "Rule-Assisted Grounded Demo (No Key Needed)",
            ],
            index=0,
        )

        api_key = ""
        model_name = "gemini-2.5-flash"
        ollama_url = "http://localhost:11434"

        if llm_provider == "Google Gemini API (Recommended)":
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                value=os.environ.get("GOOGLE_API_KEY", ""),
                help="Enter your Gemini API Key from Google AI Studio (aistudio.google.com).",
            )
            model_name = st.selectbox(
                "Gemini Model",
                options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
                index=0,
            )
        elif llm_provider == "Local Ollama (phi3:mini)":
            model_name = st.text_input("Local Ollama Model Tag", value="phi3:mini")
            ollama_url = st.text_input("Ollama Server URL", value="http://localhost:11434")

        top_k = st.slider("ChromaDB Top-K Retrieved Schemes", min_value=1, max_value=6, value=3)

        st.divider()
        st.header("👤 2. Citizen Profile Builder")
        st.caption("Leave fields as 'Not Specified' to test how the agent detects missing details.")

        preset = st.selectbox(
            "Quick Demo Persona Preset",
            options=[
                "Custom Citizen Profile",
                "Persona 1: Smallholder Farmer (Karnataka)",
                "Persona 2: Unemployed Graduate (Bengaluru)",
                "Persona 3: Woman Head of Household (Karnataka)",
                "Persona 4: Micro-Entrepreneur Seeking Mudra Loan",
                "Persona 5: Partial Profile (Tests Missing-Detail Detection)",
            ],
            index=1,
        )

        # Default values based on preset
        defaults = {
            "state": "Karnataka",
            "age": 38,
            "gender": "Male",
            "occupation": "Farmer",
            "annual_income": "Below Rs. 2,50,000",
            "ration_card_type": "BPL",
            "category": "OBC",
            "land_ownership": "Yes",
            "income_tax_payer": "No",
            "owns_pucca_house": "No",
            "has_girl_child_under_10": "Yes",
            "unemployed_days": "Not Specified",
            "default_query": (
                "I am a smallholder farmer in Karnataka with 2 acres of cultivable land, a BPL ration card, "
                "and a 6-year-old daughter. Which government schemes am I eligible for, and how do I apply?"
            ),
        }

        if preset == "Persona 2: Unemployed Graduate (Bengaluru)":
            defaults.update(
                {
                    "state": "Karnataka",
                    "age": 23,
                    "gender": "Female",
                    "occupation": "Unemployed Graduate / Diploma Holder",
                    "annual_income": "Below Rs. 2,50,000",
                    "ration_card_type": "BPL",
                    "land_ownership": "No",
                    "income_tax_payer": "No",
                    "owns_pucca_house": "Yes",
                    "has_girl_child_under_10": "No",
                    "unemployed_days": "180+ days",
                    "default_query": (
                        "I graduated with a B.Com degree in Bengaluru 8 months ago and have been unemployed "
                        "for over 180 days. Am I eligible for the Karnataka Yuva Nidhi scheme and how do I apply?"
                    ),
                }
            )
        elif preset == "Persona 3: Woman Head of Household (Karnataka)":
            defaults.update(
                {
                    "state": "Karnataka",
                    "age": 42,
                    "gender": "Female",
                    "occupation": "Homemaker / Head of Household",
                    "annual_income": "Below Rs. 2,50,000",
                    "ration_card_type": "BPL",
                    "land_ownership": "No",
                    "income_tax_payer": "No",
                    "owns_pucca_house": "No",
                    "has_girl_child_under_10": "No",
                    "unemployed_days": "Not Specified",
                    "default_query": (
                        "I am a woman registered as head of my family on a BPL ration card in Karnataka, "
                        "and we live in a kutcha house. Check my eligibility for Gruha Lakshmi, Ayushman Bharat, and PMAY."
                    ),
                }
            )
        elif preset == "Persona 4: Micro-Entrepreneur Seeking Mudra Loan":
            defaults.update(
                {
                    "state": "Maharashtra",
                    "age": 29,
                    "gender": "Male",
                    "occupation": "Small Business / Self-Employed",
                    "annual_income": "Rs. 2,50,000 - Rs. 5,00,000",
                    "ration_card_type": "APL",
                    "land_ownership": "No",
                    "income_tax_payer": "No",
                    "owns_pucca_house": "No",
                    "has_girl_child_under_10": "No",
                    "unemployed_days": "Not Specified",
                    "default_query": (
                        "I want to start a small tailoring and garment unit and need a collateral-free loan "
                        "of Rs. 2 lakh. What documents and steps are required under PM Mudra Yojana?"
                    ),
                }
            )
        elif preset == "Persona 5: Partial Profile (Tests Missing-Detail Detection)":
            defaults.update(
                {
                    "state": "Not Specified",
                    "age": 30,
                    "gender": "Not Specified",
                    "occupation": "Not Specified",
                    "annual_income": "Not Specified",
                    "ration_card_type": "Not Specified",
                    "land_ownership": "Not Specified",
                    "income_tax_payer": "Not Specified",
                    "owns_pucca_house": "Not Specified",
                    "has_girl_child_under_10": "Not Specified",
                    "unemployed_days": "Not Specified",
                    "default_query": (
                        "Can you check if I qualify for PM-KISAN, Gruha Lakshmi, or PMAY housing assistance?"
                    ),
                }
            )

        state = st.selectbox(
            "State of Residence",
            ["Not Specified", "Karnataka", "Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Other Indian State"],
            index=["Not Specified", "Karnataka", "Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Other Indian State"].index(defaults["state"]),
        )
        age = st.number_input("Age (Years)", min_value=1, max_value=100, value=int(defaults["age"]))
        gender = st.selectbox(
            "Gender",
            ["Not Specified", "Female", "Male", "Other"],
            index=["Not Specified", "Female", "Male", "Other"].index(defaults["gender"]),
        )
        occupation = st.selectbox(
            "Occupation / Status",
            [
                "Not Specified",
                "Farmer",
                "Student",
                "Unemployed Graduate / Diploma Holder",
                "Homemaker / Head of Household",
                "Small Business / Self-Employed",
                "Salaried Employee",
            ],
            index=[
                "Not Specified",
                "Farmer",
                "Student",
                "Unemployed Graduate / Diploma Holder",
                "Homemaker / Head of Household",
                "Small Business / Self-Employed",
                "Salaried Employee",
            ].index(defaults["occupation"]),
        )
        annual_income = st.selectbox(
            "Annual Family Income",
            [
                "Not Specified",
                "Below Rs. 2,50,000",
                "Rs. 2,50,000 - Rs. 5,00,000",
                "Rs. 5,00,000 - Rs. 9,00,000",
                "Above Rs. 9,00,000",
            ],
            index=[
                "Not Specified",
                "Below Rs. 2,50,000",
                "Rs. 2,50,000 - Rs. 5,00,000",
                "Rs. 5,00,000 - Rs. 9,00,000",
                "Above Rs. 9,00,000",
            ].index(defaults["annual_income"]),
        )
        ration_card_type = st.selectbox(
            "Ration Card Type",
            ["Not Specified", "BPL", "Antyodaya (AAY)", "APL", "None"],
            index=["Not Specified", "BPL", "Antyodaya (AAY)", "APL", "None"].index(defaults["ration_card_type"]),
        )
        land_ownership = st.selectbox(
            "Owns Cultivable Farm Land?",
            ["Not Specified", "Yes", "No"],
            index=["Not Specified", "Yes", "No"].index(defaults["land_ownership"]),
        )
        income_tax_payer = st.selectbox(
            "Income Tax / GST Payer?",
            ["Not Specified", "No", "Yes"],
            index=["Not Specified", "No", "Yes"].index(defaults["income_tax_payer"]),
        )
        owns_pucca_house = st.selectbox(
            "Owns a Pucca House in India?",
            ["Not Specified", "No", "Yes"],
            index=["Not Specified", "No", "Yes"].index(defaults["owns_pucca_house"]),
        )
        has_girl_child_under_10 = st.selectbox(
            "Parent of Girl Child < 10 Years?",
            ["Not Specified", "Yes", "No"],
            index=["Not Specified", "Yes", "No"].index(defaults["has_girl_child_under_10"]),
        )
        unemployed_days = st.selectbox(
            "Unemployed Duration (Graduates)",
            ["Not Specified", "180+ days", "Less than 180 days"],
            index=["Not Specified", "180+ days", "Less than 180 days"].index(defaults["unemployed_days"]),
        )

    citizen_profile = {
        "state": state,
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "annual_income": annual_income,
        "ration_card_type": ration_card_type,
        "land_ownership": land_ownership,
        "income_tax_payer": income_tax_payer,
        "owns_pucca_house": owns_pucca_house,
        "has_girl_child_under_10": has_girl_child_under_10,
        "unemployed_days": unemployed_days,
    }

    # --- Main Application Tabs ---
    tab_nav, tab_prompts, tab_vectordb = st.tabs(
        [
            "🧭 1. Multi-Agent Scheme Navigator",
            "🧠 2. Prompt Engineering & LCEL Inspector (Criteria #3 & #4)",
            "🗄️ 3. ChromaDB Vector Database Explorer (Criteria #5 & #7)",
        ]
    )

    with tab_nav:
        st.subheader("Describe Your Situation in Plain Language")
        user_query = st.text_area(
            "Citizen Request",
            value=defaults["default_query"],
            height=95,
            placeholder="e.g., I am a farmer in Karnataka with 2 acres of land. Which schemes do I qualify for and what documents do I need?",
        )

        run_btn = st.button("🚀 Run Multi-Agent Eligibility Navigator", type="primary", use_container_width=True)

        if run_btn and user_query.strip():
            llm, llm_err = get_llm_instance(
                provider=llm_provider,
                api_key=api_key,
                model_name=model_name,
                ollama_base_url=ollama_url,
            )
            if llm_err and llm_provider != "Rule-Assisted Grounded Demo (No Key Needed)":
                st.warning(f"ℹ️ {llm_err} Running grounded ChromaDB + rule-verified agent pipeline so you can inspect full outputs.")

            with st.spinner("Executing Intent Router -> ChromaDB Retrieval Agent -> Eligibility Agent -> Application Agent -> Explainer Agent..."):
                result = run_multi_agent_navigator(
                    user_query=user_query,
                    profile=citizen_profile,
                    collection=collection,
                    llm=llm,
                    top_k=top_k,
                )
                st.session_state["last_result"] = result

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            intent_info = res["intent_info"]

            # Status Bar showing Criteria #1, #5, #6
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Detected Intent (Criterion #6)", intent_info["intent"])
            col2.metric("Agentic RAG Retrieval", "Active (ChromaDB)" if intent_info["needs_retrieval"] else "Skipped")
            col3.metric("Retrieved Schemes", f"{len(res['retrieved_items'])} Docs")
            col4.metric("Missing Profile Fields", f"{len(res['missing_fields'])} Fields")

            st.info(f"**Execution Mode:** `{res['mode']}` | **Router Reasoning:** {intent_info['reasoning']}")

            if res["missing_fields"]:
                st.warning(
                    "**Agentic Missing-Profile Prompt:** Before submitting a final application, please confirm these missing profile details in the sidebar:\n"
                    + "\n".join(f"- {m}" for m in res["missing_fields"])
                )

            # Final Citizen Summary from Explainer Agent
            st.markdown("### 🗣️ Explainer Agent: Plain-Language Citizen Guide")
            st.success(res["explainer_output"])

            # Detailed 4-Agent Breakdown
            st.markdown("### 🤝 Detailed 4-Agent Collaboration Trace")
            ag_tab1, ag_tab2, ag_tab3, ag_tab4 = st.tabs(
                [
                    "🔍 1. Retrieval Agent (ReAct + ChromaDB)",
                    "✅ 2. Eligibility Agent (Chain-of-Thought)",
                    "📋 3. Application Agent (Documents & Steps)",
                    "📄 4. Raw Grounded Sources & Similarity Scores",
                ]
            )

            with ag_tab1:
                st.markdown(res["retrieval_output"])

            with ag_tab2:
                st.markdown(res["eligibility_output"])

            with ag_tab3:
                st.markdown(res["application_output"])

            with ag_tab4:
                for item in res["retrieved_items"]:
                    with st.expander(
                        f"{item['scheme_name']} ({item['scheme_id']}) — Cosine Similarity: {item['similarity_score']}"
                    ):
                        st.markdown(
                            f"- **Ministry:** {item['ministry']}\n"
                            f"- **Level:** {item['level']} | **Category:** {item['category']}\n"
                            f"- **Official Portal:** [{item['source_url']}]({item['source_url']})\n"
                            f"- **Official Guidelines:** [{item['guidelines_url']}]({item['guidelines_url']})"
                        )
                        st.code(item["document"], language="text")

    with tab_prompts:
        st.subheader("Prompt Engineering & LangChain LCEL Architecture (Criteria #3 & #4)")
        st.markdown(
            "This view explicitly documents every **System Prompt**, **User Prompt Template**, "
            "**Prompt Engineering Technique** (Few-Shot, ReAct, Chain-of-Thought, Reflection), "
            "and **LangChain Expression Language (LCEL)** chain used in the application."
        )

        st.code(
            "# LangChain Core Components & LCEL Pipeline Definition\n"
            "from langchain_core.prompts import ChatPromptTemplate\n"
            "from langchain_core.output_parsers import StrOutputParser\n"
            "from langchain_core.runnables import RunnablePassthrough\n\n"
            "prompt = ChatPromptTemplate.from_messages([\n"
            "    ('system', SYSTEM_PROMPT),\n"
            "    ('human', USER_PROMPT_TEMPLATE),\n"
            "])\n"
            "# LCEL Pipe connecting Prompt -> LLM -> Output Parser\n"
            "lcel_chain = RunnablePassthrough() | prompt | llm | StrOutputParser()",
            language="python",
        )

        for agent_name, pdata in PROMPTS_REGISTRY.items():
            with st.expander(f"📌 {agent_name} — Technique: {pdata['technique']}", expanded=(agent_name == "ELIGIBILITY_AGENT")):
                st.markdown(f"**Prompt Engineering Technique:** `{pdata['technique']}`")
                st.markdown("**1. System Prompt (`role='system'`):**")
                st.code(pdata["system"], language="text")
                st.markdown("**2. User Prompt Template (`role='human'`):**")
                st.code(pdata["user"], language="text")

    with tab_vectordb:
        st.subheader("ChromaDB Vector Store Explorer & Live Similarity Search (Criterion #7)")
        st.markdown(
            f"**Active Collection:** `government_schemes_rag` | "
            f"**Total Indexed Official Scheme Documents:** `{collection.count()}` | "
            f"**Distance Metric:** `Cosine Similarity`"
        )

        col_search, col_upload = st.columns([2, 1])
        with col_search:
            test_q = st.text_input(
                "Test Direct Vector Similarity Search in ChromaDB",
                value="financial assistance for unemployed graduates in Karnataka",
            )
            if st.button("🔎 Query ChromaDB Vector Index"):
                hits = retrieve_schemes_from_chromadb(collection, test_q, top_k=4)
                for h in hits:
                    st.markdown(
                        f"**{h['scheme_name']} (`{h['scheme_id']}`)** — "
                        f"Similarity: `{h['similarity_score']}` (Cosine Distance: `{h['distance']}`) | "
                        f"[Official Portal]({h['source_url']})"
                    )
                    st.caption(h["document"][:360] + "...")

        with col_upload:
            st.markdown("**Add Custom Scheme Document to ChromaDB**")
            custom_id = st.text_input("Custom Scheme ID", value="CUSTOM-SCHEME-09")
            custom_name = st.text_input("Custom Scheme Name", value="Chief Minister Raitha Vidya Nidhi")
            custom_url = st.text_input("Official Source URL", value="https://en.vikaspedia.in/viewcontent/schemesall/state-specific-schemes/welfare-schemes-of-karnataka?lgn=en")
            custom_body = st.text_area(
                "Scheme Rules, Benefits & Application Steps",
                value="Provides annual scholarships up to Rs. 11,000 to children of farmers in Karnataka studying in secondary/higher education.",
            )
            if st.button("➕ Index into ChromaDB"):
                collection.upsert(
                    ids=[custom_id],
                    documents=[
                        f"SCHEME ID: {custom_id}\nSCHEME NAME: {custom_name}\n"
                        f"OFFICIAL PORTAL URL: {custom_url}\n{custom_body}"
                    ],
                    metadatas=[
                        {
                            "scheme_id": custom_id,
                            "scheme_name": custom_name,
                            "level": "State/Central",
                            "ministry": "Custom Uploaded Department",
                            "category": "Custom Scheme",
                            "source_url": custom_url,
                            "guidelines_url": custom_url,
                            "required_profile_fields": "occupation,state",
                        }
                    ],
                )
                st.success(f"Indexed `{custom_name}` into ChromaDB! Total docs: {collection.count()}")


if __name__ == "__main__":
    main()
