import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
pick = lambda xs: xs[rng.integers(len(xs))]

TITLES = ["Software Engineer", "Data Analyst", "Marketing Manager", "Customer Support Agent",
          "Accountant", "Sales Executive", "Graphic Designer", "HR Coordinator", "Project Manager",
          "Operations Associate", "Content Writer", "Mechanical Engineer", "Nurse", "Teacher"]
COMPANIES = ["Northwind Systems", "BrightPath Solutions", "Orbit Analytics", "Greenfield Group",
             "Summit Logistics", "Bluewave Media", "Apex Health", "Cedar Financial"]
SKILLS = ["Python", "SQL", "Excel", "communication", "project planning", "Figma", "Java",
          "stakeholder management", "budgeting", "customer service", "data analysis", "AWS"]
RESP = ["Collaborate with cross-functional teams to deliver quarterly goals.",
        "Prepare weekly reports and present insights to leadership.",
        "Maintain documentation and ensure quality standards are met.",
        "Support day-to-day operations and improve existing processes.",
        "Work closely with clients to understand and resolve their needs.",
        "Participate in planning, code reviews and team retrospectives."]
BENEFITS = ["health insurance", "paid leave", "learning budget", "flexible hours", "annual bonus",
            "retirement plan"]
LEGIT_NOISE = ["Remote work available.", "Immediate start for the right candidate.",
               "Competitive salary.", "Fast-paced environment.", "Apply today."]

SCAM_OBVIOUS = [
    "Earn $800 per week working from home! No experience needed.",
    "URGENT hiring! Limited slots, apply immediately and start earning today.",
    "Guaranteed income, be your own boss. Send us a message on WhatsApp to begin.",
    "A small registration fee is required to start your training.",
    "Send your bank details so we can set up your payments. Easy money!",
    "You will purchase your own equipment kit and be reimbursed after the first month.",
    "Interview will be held on Telegram. Contact hr.careers.team@gmail.com",
    "Pay a security deposit to secure your position. Act now!",
]
SCAM_SUBTLE = [
    "Immediate start. Interviews on Telegram.",
    "Equipment must be purchased first and reimbursed after your first week.",
    "Send your resume and ID copy to recruit.now@yahoo.com",
    "Limited positions, apply immediately.",
    "Flexible, fully remote role with weekly payouts of $600 per week.",
    "Training fee applies and is refunded later.",
]


def legit_text(fake_like=False):
    desc = f"We are looking for a {pick(TITLES)} with strong {pick(SKILLS)} and {pick(SKILLS)} skills. "
    desc += " ".join(rng.choice(RESP, 3, replace=False))
    req = f"Experience with {pick(SKILLS)} and {pick(SKILLS)}. {int(rng.integers(1, 8))}+ years of relevant experience."
    desc += f" We offer {pick(BENEFITS)} and {pick(BENEFITS)}."
    if rng.random() < 0.2:
        desc += " " + pick(LEGIT_NOISE)
    return desc, req


def make(n=6000, fake_rate=0.08):
    rows = []
    for _ in range(n):
        fake = rng.random() < fake_rate
        title = pick(TITLES)
        company = pick(COMPANIES)
        desc, req = legit_text()
        if fake:
            if rng.random() < 0.4:      # obvious scam
                desc = " ".join(rng.choice(SCAM_OBVIOUS, rng.integers(2, 4), replace=False)) + " " + desc[:80]
                req = "No special requirements. Anyone can apply."
                company_profile = "" if rng.random() < 0.7 else f"{company} is a growing company."
            else:                        # subtle scam: professional text + 1-2 red flags
                desc += " " + " ".join(rng.choice(SCAM_SUBTLE, rng.integers(1, 3), replace=False))
                company_profile = f"{company} is a leading provider of innovative solutions." if rng.random() < 0.6 else ""
        else:
            company_profile = f"{company} is an established firm with {int(rng.integers(20, 5000))} employees."
        rows.append(dict(
            title=title, company_profile=company_profile, description=desc, requirements=req,
            has_company_logo=int(rng.random() < (0.35 if fake else 0.85)),
            has_company_profile=int(bool(company_profile)),
            telecommuting=int(rng.random() < (0.55 if fake else 0.2)),
            salary_given=int(rng.random() < (0.2 if fake else 0.45)),
            free_email_contact=int(rng.random() < (0.45 if fake else 0.06)),
            education_listed=int(rng.random() < (0.35 if fake else 0.7)),
            description_length=len(desc),
            fraudulent=int(fake),
        ))
    df = pd.DataFrame(rows)
    flip = rng.random(len(df)) < 0.01     # 1% label noise, like real-world mislabeled data
    df.loc[flip, "fraudulent"] = 1 - df.loc[flip, "fraudulent"]
    return df


if __name__ == "__main__":
    d = make()
    d.to_csv("data/jobs.csv", index=False)
    print(f"Saved data/jobs.csv ({len(d)} rows, fake rate {d.fraudulent.mean():.1%})")
