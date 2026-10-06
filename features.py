import re
import pandas as pd

NUMERIC = [
    "has_company_logo", "has_company_profile", "telecommuting", "salary_given",
    "free_email_contact", "education_listed", "description_length",
]

LABELS = {
    "has_company_logo": "Company logo present",
    "has_company_profile": "Company profile present",
    "telecommuting": "Remote/telecommuting",
    "salary_given": "Salary range given",
    "free_email_contact": "Contact via free email (gmail/yahoo...)",
    "education_listed": "Education requirement listed",
    "description_length": "Description length",
}

RED_FLAGS = {
    r"registration fee|processing fee|training fee|pay .{0,20}(fee|deposit)|security deposit":
        "Asks for money upfront",
    r"no experience (needed|required|necessary)": "'No experience needed' for a paid role",
    r"whatsapp|telegram|text us at": "Interviews/contact via messaging apps",
    r"\$\s?\d{3,5}\s?(per|a|/)\s?(day|week)": "Unrealistic pay per day/week",
    r"wire transfer|western union|bank details|bank account (number|details)|ssn|social security":
        "Requests sensitive financial/personal data",
    r"urgent(ly)? hiring|apply immediately|limited (slots|positions)|act now":
        "High-pressure urgency language",
    r"@(gmail|yahoo|hotmail|outlook)\.com": "Free personal email used for hiring",
    r"guaranteed (income|salary|job)|easy money|be your own boss": "Too-good-to-be-true promises",
    r"purchase (your own )?(equipment|kit|laptop)|reimburse(d)? after": "Asks you to buy equipment first",
}


def build_text(df: pd.DataFrame) -> pd.Series:
    cols = ["title", "company_profile", "description", "requirements"]
    return df[cols].fillna("").agg(" ".join, axis=1)


def find_red_flags(text: str):
    return [msg for pat, msg in RED_FLAGS.items() if re.search(pat, text, re.I)]
