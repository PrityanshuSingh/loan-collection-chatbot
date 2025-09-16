# src/strategy_playbook.py

strategy_playbook = {
    'Cooperative Optimizers': {
        "goal": "Secure a full, on-time payment while reinforcing the customer's positive behavior and retaining their goodwill.",
        "psychology_levers": ["Convenience", "Positive Reinforcement", "Consistency"],
        "default_tone": "Positive, Appreciative, and Professional",
        "opening_line": "Thank you for your consistent track record. Just a friendly reminder that your payment of {due_amount} is due. Paying now helps maintain your excellent history.",
        "ongoing_global_tactics": "Keep communication minimal and efficient. Proactively offer help or information if they seem hesitant. The goal is a frictionless payment experience.",
        "disengagement_protocol": "Of course. I've noted your request. We'll mark this as resolved for now. Have a great day!",
        "recovery_guidelines_applied": ["Fair Practices Code"],
        "cta_options": ["Pay now in 2 taps", "Set up autopay to save time", "Choose a different payment date"],
        "contingency_cta_options": ["Can I provide your statement details?", "Would you like to know the benefits of autopay?"],
        "escalation_trigger": "If two consecutive payments are missed, their behavior has changed. Re-evaluate and apply the 'Willing but Struggling' strategy.",
        "closure_line": "Thank you! Your payment is confirmed. We appreciate you staying on track."
    },
    'Willing but Struggling': {
        "goal": "Secure a partial payment to halt fees and guide the user into a formal, manageable repayment plan. The key is converting intent into action.",
        "psychology_levers": ["Empathy", "Reciprocity", "Flexible Options", "Hope"],
        "default_tone": "Supportive, Collaborative, and Non-judgmental",
        "opening_line": "Hi there. We see you have an overdue amount of {due_amount} and understand things can be challenging. Let's work together to find a flexible solution. Even a small payment today can prevent extra fees and get things back on track.",
        "ongoing_global_tactics": "Acknowledge their financial difficulty first. If they reject an amount, pivot to questions like, 'I understand that's not possible. What would a manageable first step look like for you?' Frame every option as a way to help them.",
        "disengagement_protocol": "I understand and I'm sorry to have upset you. I will end our chat now. Please know that we are here to help whenever you are ready.",
        "recovery_guidelines_applied": ["Borrower in Temporary Distress", "Structured Debt Settlement"],
        "cta_options": ["Make a partial payment now", "Split overdue into 2 payments", "Request a 7-day payment holiday"],
        "contingency_cta_options": ["Can we explore a plan that fits your current situation?", "Would it help to see how a small payment reduces future fees?"],
        "escalation_trigger": "Two broken Promises-to-Pay (PTP) or no engagement after three attempts signals a need for a more formal channel.",
        "closure_line": "Thank you for setting up this plan. We've scheduled your reminders. Please don't hesitate to reach out if your situation changes—we're here to help."
    },
    'Passive Defaulters': {
        "goal": "Create urgency and secure a payment (full or minimum) by clearly linking inaction to the negative consequence on their credit score.",
        "psychology_levers": ["Loss Aversion", "Urgency", "Scarcity (of time)"],
        "default_tone": "Firm, Factual, and Professional",
        "opening_line": "Important reminder for your loan account. Your payment of {due_amount} is now {dpd} days overdue. To prevent a negative entry on your CIBIL report, payment of at least the minimum amount is required by [Date].",
        "ongoing_global_tactics": "Be direct and concise. If the user deflects with excuses, gently but firmly return to the credit score consequence. For example: 'I understand, but my priority is to help you prevent a negative credit report. Shall we proceed with the minimum payment?'",
        "disengagement_protocol": "Understood. Please be aware that continued non-payment will likely result in negative credit reporting as per our policy. We will cease digital messages now.",
        "recovery_guidelines_applied": ["Notice of Default", "Fair Practices Code"],
        "cta_options": ["Pay full amount", "Pay minimum to protect credit score", "Understand the credit reporting process"],
        "contingency_cta_options": ["View your statement", "See the potential impact on your credit score?"],
        "escalation_trigger": "No payment or meaningful contact for 21 days should trigger a formal pre-notice communication.",
        "closure_line": "Payment received. Your account is now current. Please ensure future payments are on time to maintain a healthy credit history."
    },
    'Stressed Resistors': {
        "goal": "De-escalate the user's frustration by acknowledging their complaint, then pivot to a resolution path. Payment is a secondary goal until the primary issue is addressed.",
        "psychology_levers": ["De-escalation", "Acknowledgment", "Fairness", "Procedural Justice"],
        "default_tone": "Calm, Respectful, and Resolution-Oriented",
        "opening_line": "We see you've raised a concern and we take this very seriously. A complaint has been logged (Ref: [Ticket ID]). My priority is to understand and document your issue for our review team. Please know your payment obligation continues while we investigate.",
        "ongoing_global_tactics": "Do not mention payment again until their complaint has been fully heard and acknowledged. Use active listening phrases like 'So, to confirm, the issue is...' The goal is to make them feel heard, which reduces hostility.",
        "disengagement_protocol": "I understand your frustration. To ensure this is handled correctly, I am escalating your case [Ticket ID] to our specialist grievance team. They will contact you via a formal channel.",
        "recovery_guidelines_applied": ["Grievance Redressal", "Harassment Prevention"],
        "cta_options": ["Track complaint status", "Upload documents for your case", "Schedule a call with a supervisor"],
        "contingency_cta_options": ["What information can I provide to help with your concern?", "Is there a specific detail I can add to the case notes for you?"],
        "escalation_trigger": "Refusal to cooperate after their complaint has been formally resolved triggers a pre-litigation review.",
        "closure_line": "Thank you. Your information has been documented and attached to case [Ticket ID]. Our team will provide a formal update within 72 business hours."
    },
    'High-Risk Avoiders': {
        "goal": "Break the pattern of avoidance by securing a micro-payment. This re-establishes a payment habit and pauses imminent formal escalation.",
        "psychology_levers": ["Small Wins", "Relief Framing", "Urgency"],
        "default_tone": "Empathetic but Direct and Firm",
        "opening_line": "We understand things might be difficult. This is a final attempt to resolve your overdue account before it is escalated for formal recovery. A small payment today can pause this process and show your intent to resolve this.",
        "ongoing_global_tactics": "The focus is on action, not conversation. If they reject a payment, pivot immediately to a smaller action. 'If not that amount, what is a realistic first step you can take today?' If they deflect, gently bring it back: 'I hear you, but taking a small step now is the only way to pause the escalation.'",
        "disengagement_protocol": "Understood. We will cease digital contact. Please be aware that your account is now pending formal review and escalation due to its overdue status.",
        "recovery_guidelines_applied": ["Harassment Prevention", "Structured Debt Settlement"],
        "cta_options": ["Discuss a settlement offer", "See payment plan options"],
        "contingency_cta_options": ["Would you like to understand what 'formal recovery' means?", "Is there a smaller amount you can pay right now to show intent?"],
        "escalation_trigger": "Zero payment after the communication burst is an objective trigger for legal review.",
        "closure_line": "Thank you. Your payment is received. This has paused the formal escalation. A specialist will contact you within 48 hours to discuss a permanent plan."
    }
}