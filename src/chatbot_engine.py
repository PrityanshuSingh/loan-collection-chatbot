# src/chatbot_engine.py

import pandas as pd
import numpy as np
import json
import re
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from groq import Groq
import warnings

# --- LangChain RAG Imports (only for type hinting, actual objects passed in) ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- Custom App Imports ---
from config import MAX_NEGATIVE_STRIKES

warnings.filterwarnings('ignore')

class IntegratedChatbot:
    """
    An advanced chatbot engine that integrates:
    1. A predictive XGBoost model for default probability.
    2. A KMeans clustering model for customer persona identification.
    3. A Retrieval-Augmented Generation (RAG) system for compliance.
    4. A Large Language Model (Groq) for conversational intelligence.
    5. A detailed strategy playbook for persona-based interaction.
    """
    def __init__(self, predictive_assets, persona_assets, playbook, groq_client, groq_model_name, retriever):
        self.pred_assets = predictive_assets
        self.pers_assets = persona_assets
        self.playbook = playbook
        self.groq_client = groq_client
        self.groq_model_name = groq_model_name
        self.retriever = retriever
        self.memory = ConversationBufferMemory(return_messages=True)
        self.reset_state()

    def reset_state(self):
        """Resets the chatbot's state for a new conversation."""
        self.persona = None
        self.customer_data = None
        self.default_prob = 0.0
        self.persona_definition = {}
        self.financial_details = {}
        self.offers_made = set()
        self.turn_count = 0
        self.conversation_ended = False
        self.prediction_features = {}
        self.persona_features = {}
        self.negative_sentiment_strikes = 0
        self.max_negative_strikes = MAX_NEGATIVE_STRIKES
        self.success_metrics = {
            "last_turn_score": 0,  # per turn
            "cumulative_score": 0  # cumulative score of the conversation
        }
        self.last_strategy_used = None
        self.memory.clear()

    def _get_rag_context(self, query: str) -> str:
        """Retrieves relevant context from the knowledge base using RAG."""
        if not self.retriever:
            return "No knowledge base available."
        try:
            relevant_docs = self.retriever.get_relevant_documents(query)
            context = "\n---\n".join([doc.page_content for doc in relevant_docs])
            return context
        except Exception as e:
            print(f"[RAG ERROR] Failed to retrieve documents: {e}")
            return "Error retrieving guidelines."

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Ensures a DataFrame has all required columns in the correct order."""
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0
        return out[cols] # Return with columns in the specified order

    def _engineer_for_prediction(self, df):
        """Applies the exact same feature engineering as the predictive model training."""
        df = df.copy()
        le = self.pred_assets['label_encoders']
        for col in ['Location', 'EmploymentStatus', 'LoanType']:
            if col in df.columns:
                known_classes = set(le[col].classes_)
                # Handle unseen values during prediction by mapping them to a new 'Other' category
                df[col] = df[col].apply(lambda x: x if x in known_classes else 'Other')
                if 'Other' not in le[col].classes_:
                    le[col].classes_ = np.append(le[col].classes_, 'Other')
                df[col] = le[col].transform(df[col])

        df['AgeRiskBucket'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3])
        r = df['InterestRate'] / (12 * 100)
        n = df['TenureMonths']
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            df['EMI'] = (df['LoanAmount'] * r * (1 + r).pow(n)) / ((1 + r).pow(n) - 1)
        df['DTI'] = df['EMI'] / (df['Income'] / 12)
        df['LTI'] = df['LoanAmount'] / df['Income']
        total_interest = (df['EMI'] * df['TenureMonths']) - df['LoanAmount']
        df['InterestBurden'] = total_interest.clip(lower=0) / df['TenureMonths'] / (df['Income'] / 12)
        df['TAB'] = df['LoanAmount'] / df['TenureMonths']
        df['MPR'] = df['MissedPayments'] / df['TenureMonths']
        df['AvgDelay'] = df['DelaysDays'] / (df['MissedPayments'] + 1)
        df['PPR'] = df['PartialPayments'] / df['TenureMonths']
        df['BRI'] = (0.5 * df['MPR']) + (0.3 * (df['DelaysDays'] / df['TenureMonths'])) + (0.2 * df['PPR'])
        df['ES'] = 1 / (1 + df['ResponseTimeHours'])
        df['DES'] = (df['AppUsageFrequency'] + (df['WebsiteVisits'] / 30)) / 2
        df['IFI'] = df['InteractionAttempts'] / df['TenureMonths']
        df['SentimentRiskScore'] = (1 - df['SentimentScore']) * df['IFI']
        df['Has_High_Complaints'] = (df.get('Complaints', 0) > 2).astype(int)
        df['Complaints_x_HighDTI'] = df['Has_High_Complaints'] * (df['DTI'] > self.pred_assets['high_dti_threshold']).astype(int)
        df['Complaints_x_HighBRI'] = df['Has_High_Complaints'] * (df['BRI'] > self.pred_assets['high_bri_threshold']).astype(int)
        df['HighBRI_x_LowDES'] = (df['BRI'] > self.pred_assets['high_bri_threshold']).astype(int) * (df['DES'] < self.pred_assets['low_des_threshold']).astype(int)
        df['DTI_x_MissedPayments'] = df['DTI'] * df['MissedPayments']
        df['LoanAmount_x_InterestRate'] = df['LoanAmount'] * df['InterestRate']
        df.drop(columns=['Complaints'], inplace=True, errors='ignore')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def _engineer_for_persona(self, df):
        """Applies the exact same feature engineering as the persona model training."""
        df = df.copy()
        df['DebtToIncomeRatio'] = df['LoanAmount'] / (df['Income'] + 1)
        df['MonthlyPaymentBurden'] = (df['LoanAmount'] * (1 + df['InterestRate']/100)) / df['TenureMonths']
        df['PaymentToIncomeRatio'] = df['MonthlyPaymentBurden'] / (df['Income']/12 + 1)
        df['InterestPremium'] = df['InterestRate'] - self.pers_assets['median_interest']
        df['PaymentConsistency'] = 1 / (1 + df['MissedPayments'])
        df['DelayIntensity'] = df['DelaysDays'] / (df['TenureMonths'] * 30 + 1)
        df['PartialPaymentRate'] = df['PartialPayments'] / (df['MissedPayments'] + 1)
        df['ResponseRate'] = 1 / (1 + df['ResponseTimeHours']/24)
        df['InteractionIntensity'] = df['InteractionAttempts'] / (df['TenureMonths'] + 1)
        df['DigitalEngagement'] = (df['AppUsageFrequency'] + df['WebsiteVisits']/30) / 2
        df['SentimentNormalized'] = (df['SentimentScore'] + 1) / 2
        df['ComplaintRate'] = df.get('Complaints', 0) / (df['InteractionAttempts'] + 1)
        df['EmploymentRisk'] = df['EmploymentStatus'].map(self.pers_assets['employment_risk_map']).fillna(0.5)
        df['AgeRisk'] = np.where(df['Age'] < 25, 0.8, np.where(df['Age'] > 60, 0.7, 0.4))
        df['EarlyDefaultRisk'] = np.where(df['TenureMonths'] < 6, 1.0, np.where(df['TenureMonths'] < 12, 0.7, 0.4))
        df['FinancialCapacity'] = ((1-df['DebtToIncomeRatio'])*0.4 + (1-df['PaymentToIncomeRatio'])*0.3 + (1-df['InterestPremium'])*0.15 + (1-df['EmploymentRisk'])*0.15)
        df['PaymentReliability'] = (df['PaymentConsistency']*0.4 + (1-df['DelayIntensity'])*0.3 + df['PartialPaymentRate']*0.2 + (1-df['EarlyDefaultRisk'])*0.1)
        df['EngagementQuality'] = (df['ResponseRate']*0.3 + df['InteractionIntensity']*0.2 + df['DigitalEngagement']*0.3 + df['SentimentNormalized']*0.2)
        df['CooperationLevel'] = (df['SentimentNormalized']*0.4 + (1-df['ComplaintRate'])*0.3 + df['ResponseRate']*0.3)
        df['OverallRiskScore'] = ((1-df['FinancialCapacity'])*0.35 + (1-df['PaymentReliability'])*0.35 + (1-df['EngagementQuality'])*0.15 + (1-df['CooperationLevel'])*0.15)
        return df

    def _calculate_financials(self, customer_data):
        """Calculates key financial details for the conversation."""
        p = customer_data.get('LoanAmount', 0)
        i = customer_data.get('InterestRate', 0)
        n = customer_data.get('TenureMonths', 1)
        r = i / (12 * 100)
        missed_payments = customer_data.get('MissedPayments', 0)
        emi = (p * r * (1 + r)**n) / ((1 + r)**n - 1) if r > 0 else (p/n if n > 0 else 0)
        total_overdue = emi * missed_payments
        late_fees = total_overdue * 0.05
        self.financial_details = {"Total Amount Due Now": f"₹{total_overdue + late_fees:,.0f}"}

    def start_conversation(self, customer_data):
        """Initiates a new conversation, running both predictive and persona models."""
        self.reset_state()
        self.customer_data = customer_data
        customer_df = pd.DataFrame([customer_data])
        self._calculate_financials(customer_data)

        # --- Stage 1: Default Prediction ---
        df_pred_eng = self._engineer_for_prediction(customer_df)
        self.prediction_features = df_pred_eng.to_dict(orient='records')[0]
        model_cols = self.pred_assets['feature_names']
        df_pred_eng_aligned = self._ensure_columns(df_pred_eng, model_cols)
        X_pred_scaled = self.pred_assets['scaler'].transform(df_pred_eng_aligned)
        self.default_prob = self.pred_assets['model'].predict_proba(X_pred_scaled)[:, 1][0]
        is_defaulter = self.default_prob > 0.5

        if not is_defaulter:
            self.persona = "Non-Defaulter"
            opening_message = "Hi! This is a friendly reminder about your upcoming payment. Let me know if you need any assistance."
            self.memory.chat_memory.add_ai_message(opening_message)
            return opening_message, self.default_prob, self.persona

        # --- Stage 2: Persona Identification (only if high default risk) ---
        df_pers_eng = self._engineer_for_persona(customer_df)
        self.persona_features = df_pers_eng.to_dict(orient='records')[0]
        persona_cols = self.pers_assets['persona_feature_names']
        df_pers_ready = df_pers_eng[persona_cols].fillna(self.pers_assets['imputer_data_median'])
        X_pers_scaled = self.pers_assets['persona_scaler'].transform(df_pers_ready)
        predicted_cluster = self.pers_assets['model'].predict(X_pers_scaled)[0]
        final_persona_id = self.pers_assets['cluster_mapping'][predicted_cluster]
        self.persona_definition = self.pers_assets['definitions'][final_persona_id]
        self.persona = self.persona_definition['name']

        opening_message = self.playbook[self.persona]['opening_line'].format(
            due_amount=self.financial_details['Total Amount Due Now'],
            dpd=self.customer_data.get('DelaysDays', 'some')
        )
        self.memory.chat_memory.add_ai_message(opening_message)
        return opening_message, self.default_prob, self.persona

    def _analyze_turn_with_llm(self, history, customer_utterance, persona):
        """
        Performs a comprehensive analysis of the user's message to extract sentiment, intent, and determine strategy.
        This informs the main response generation logic.
        """
        prompt = f"""Analyze the user's latest message in a loan collection chat and determine the best strategy.
        **User Persona:** {persona}
        **Conversation History:** {history}
        **Latest User Message:** "{customer_utterance}"
        
        Based on the user's sentiment and intent, choose the most appropriate strategy from these options:
        - "Empathetic Listener" - for distressed, overwhelmed users
        - "Solution Provider" - for cooperative users seeking options
        - "Firm but Fair" - for evasive or non-committal users
        - "Information Clarifier" - for confused users asking questions
        - "Negotiation Facilitator" - for users wanting to negotiate terms
        - "Closure Specialist" - for users wanting to end the conversation
        
        Return a JSON object with your analysis:
        - "sentiment": Analyze the user's emotion (e.g., "Frustrated", "Confused", "Cooperative", "Hostile", "Apologetic").
        - "user_intent": What is the user trying to do? (e.g., "Requesting information", "Expressing inability to pay", "Negotiating payment", "Hostile disengagement").
        - "strategy": Choose the most appropriate strategy from the list above based on sentiment and intent.
        JSON:"""
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], 
                model=self.groq_model_name, 
                temperature=0.1, 
                response_format={"type": "json_object"}
            )
            return json.loads(chat_completion.choices[0].message.content.strip())
        except Exception as e:
            print(f"[LLM ANALYSIS ERROR] {e}")
            return {
                "sentiment": "Neutral", 
                "user_intent": "Unknown",
                "strategy": "Information Clarifier"
        }

    def _track_offers(self, text):
        """Finds and stores any monetary offers made by the bot to avoid repetition."""
        found_offers = re.findall(r'[₹|Rs]\.?\s*([\d,]+)', text)
        for offer in found_offers: self.offers_made.add(offer.replace(',', ''))

    def _calculate_conversation_success(self, analysis_result):
        """
        Computes conversation success score (0-100) based on:
        - Sentiment (positive/neutral vs negative)
        - Engagement (turn count)
        - Persona compliance (offers not repeated)
        Returns a per-turn score and updates the running average.
        """
        # Base score per turn
        score = 50  # base
        sentiment = analysis_result.get("sentiment", "Neutral")
        
        # Sentiment scoring
        if sentiment in ["Cooperative", "Apologetic"]:
            score += 30
        elif sentiment in ["Frustrated", "Hostile"]:
            score -= 20
        elif sentiment in ["Confused", "Questioning"]:
            score += 10

        # Engagement factor: reward continued engagement
        engagement_factor = min(self.turn_count, 10) * 2
        score += engagement_factor

        # Persona compliance: reward if offers not repeated
        if not self.offers_made:
            score += 10

        # Strike penalty
        if self.negative_sentiment_strikes > 0:
            score -= (self.negative_sentiment_strikes * 15)

        # Clamp per-turn score between 0-100
        score = max(0, min(100, score))
        self.success_metrics['last_turn_score'] = score

        # --- Update running average correctly ---
        if self.turn_count == 1:
            # First turn: cumulative = current score
            self.success_metrics['cumulative_score'] = score
        else:
            # Running average: ((previous_avg * (n-1)) + current_score) / n
            prev_avg = self.success_metrics['cumulative_score']
            n = self.turn_count
            new_avg = ((prev_avg * (n - 1)) + score) / n
            self.success_metrics['cumulative_score'] = new_avg

        return self.success_metrics['cumulative_score']


    # def get_next_response(self, customer_utterance):
    #     """Generates the next chatbot response based on the full context."""
    #     if self.conversation_ended: 
    #         return "This conversation has ended.", {}

    #     if not self.groq_client: 
    #         return "Error: Groq model is not configured.", {}

    #     self.memory.chat_memory.add_user_message(customer_utterance)
    #     self.turn_count += 1
    #     history = "\n".join([f"{'User' if 'user' in msg.type else 'Bot'}: {msg.content}" for msg in self.memory.chat_memory.messages])

    #     analysis_result = self._analyze_turn_with_llm(history, customer_utterance, self.persona)

    #     strategy_for_persona = self.playbook[self.persona]
    #     negative_sentiments = ["Frustrated", "Hostile"]
    #     disengagement_intents = ["Hostile disengagement"]

    #     # --- End chat if hostile disengagement ---
    #     if analysis_result.get("user_intent") in disengagement_intents:
    #         self.conversation_ended = True
    #         response = strategy_for_persona.get("disengagement_protocol", "I understand. I will end our chat now.")
    #         self.memory.chat_memory.add_ai_message(response)
    #         return response, analysis_result

    #     # --- Track negative sentiment strikes ---
    #     if analysis_result.get("sentiment") in negative_sentiments:
    #         self.negative_sentiment_strikes += 1
    #         if self.negative_sentiment_strikes >= self.max_negative_strikes:
    #             self.conversation_ended = True
    #             response = strategy_for_persona.get(
    #                 "disengagement_protocol",
    #                 "I understand. To avoid further frustration, I will end our chat now."
    #             )
    #             self.memory.chat_memory.add_ai_message(response)
    #             return response, analysis_result

    #     # --- Conversation Success Score (per turn) ---
    #     turn_score = self._calculate_conversation_success(analysis_result)
    #     self.success_metrics['last_turn_score'] = turn_score

    #     # ✅ Update cumulative score
    #     self.success_metrics['cumulative_score'] += turn_score

    #     # ✅ Store the strategy name correctly
    #     self.last_strategy_used = strategy_for_persona.get("name", self.persona)  # fallback to persona name

    #     # ✅ Store metrics in analysis result for sidebar
    #     analysis_result['success_score'] = turn_score
    #     analysis_result['cumulative_score'] = self.success_metrics['cumulative_score']
    #     analysis_result['applied_strategy'] = strategy_for_persona.get("name", self.persona)


    #     # --- RAG Context ---
    #     rag_context = self._get_rag_context(customer_utterance)
        
    #     # ##################################################################
    #     # ### CORE PROMPT TEMPLATE WITH EXPLICIT INR CURRENCY CONTEXT ###
    #     # ##################################################################
    #     generation_prompt_template = """
    #     You are an expert collections assistant. Your goal is a natural, empathetic, and compliant conversation to guide the user towards a resolution.

    #     **1. STRATEGY & GUIDELINES (Your instruction manual for this user)**
    #     ---
    #     {strategy_document}
    #     ---

    #     **2. CUSTOMER PROFILE & FINANCIALS (The single source of truth for all numbers)**
    #     - **Loan & Financial Details (All amounts are in Indian Rupees - INR)**: {financial_details}
    #     ---

    #     **3. PREDICTIVE MODEL INSIGHTS (Key risk indicators for default prediction)**
    #     These features explain WHY the model predicts a {default_prob:.1%} chance of default.
    #     ---
    #     {prediction_features}
    #     ---
        
    #     **4. PERSONA MODEL INSIGHTS (Key behavioral and financial drivers for persona)**
    #     These features explain WHY the user is classified as a '{persona}' persona.
    #     ---
    #     {persona_features}
    #     ---

    #     **5. REAL-TIME CONVERSATION ANALYSIS:**
    #     - Latest Customer Sentiment: {sentiment}
    #     - User's Inferred Intent: {user_intent}
    #     - **Offers Already Made/Rejected**: {offers_made}
    #     - Full Conversation History: {history}
    #     ---

    #     **6. RELEVANT GUIDELINES (from knowledge base)**
    #     Based on the user's last message, here are the most relevant internal guidelines. You MUST use these to ensure compliance.
    #     ---
    #     {rag_context}
    #     ---

    #     **YOUR TASK & STRICT RULES:**
    #     Write the next response. Follow these rules without exception:
    #     1.  **ACKNOWLEDGE AND ADVANCE:** Address the user's LAST message based on their intent, then pivot towards the `goal` defined in the strategy document.
    #     2.  **USE ALL CONTEXT:** You MUST let the "RELEVANT GUIDELINES", "PREDICTIVE INSIGHTS", and "PERSONA INSIGHTS" directly influence your word choice to be compliant, accurate, and psychologically effective.
    #     3.  **NEVER REPEAT OFFERS:** The user has already seen or rejected these amounts: {offers_made}. DO NOT suggest them again unless they ask.
    #     4.  **BE ACTION-ORIENTED:** Your response must guide the user towards a resolution. When appropriate, present one or two of the most relevant `cta_options` from the strategy document.
    #     5.  **BE CONCISE AND HUMAN:** Your response MUST be very short (1-2 sentences, max 50 words). Maintain the persona's `default_tone`.

    #     Bot Response:
    #     """
    #     prompt = PromptTemplate(
    #         template=generation_prompt_template,
    #         input_variables=[
    #             "strategy_document", "financial_details", "default_prob",
    #             "prediction_features", "persona", "persona_features",
    #             "sentiment", "user_intent", "offers_made",
    #             "history", "rag_context"
    #         ]
    #     )
        
    #     pred_features_rounded = {k: f"{v:.2f}" for k, v in self.prediction_features.items() if isinstance(v, (int, float))}
    #     pers_features_rounded = {k: f"{v:.2f}" for k, v in self.persona_features.items() if isinstance(v, (int, float))}

    #     prompt_filled = prompt.format(
    #         strategy_document=json.dumps(strategy_for_persona, indent=2),
    #         financial_details=json.dumps(self.financial_details, indent=2),
    #         default_prob=self.default_prob,
    #         prediction_features=json.dumps(pred_features_rounded, indent=2),
    #         persona=self.persona,
    #         persona_features=json.dumps(pers_features_rounded, indent=2),
    #         sentiment=analysis_result['sentiment'],
    #         user_intent=analysis_result['user_intent'],
    #         offers_made=list(self.offers_made) if self.offers_made else "None",
    #         history=history,
    #         rag_context=rag_context
    #     )

    #     try:
    #         chat_completion = self.groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt_filled}], model=self.groq_model_name, temperature=0.5)
    #         bot_response = chat_completion.choices[0].message.content
    #     except Exception as e:
    #         print(f"[LLM GENERATION ERROR] Failed: {e}")
    #         bot_response = "I'm having a technical issue at the moment. Please give me a moment."

    #     self._track_offers(bot_response)
    #     self.memory.chat_memory.add_ai_message(bot_response)
    #     return bot_response, analysis_result
    
    def get_next_response(self, customer_utterance):
        """Generates the next chatbot response based on the full context."""
        if self.conversation_ended: 
            return "This conversation has ended.", {}

        if not self.groq_client: 
            return "Error: Groq model is not configured.", {}

        # Add user message to memory
        self.memory.chat_memory.add_user_message(customer_utterance)
        self.turn_count += 1
        history = "\n".join([f"{'User' if 'user' in msg.type else 'Bot'}: {msg.content}" 
                            for msg in self.memory.chat_memory.messages])

        # --- LLM analysis per turn (now includes strategy) ---
        analysis_result = self._analyze_turn_with_llm(history, customer_utterance, self.persona)

        # --- Get strategy from persona playbook or use LLM-determined strategy ---
        strategy_for_persona = self.playbook.get(self.persona, {})
        negative_sentiments = ["Frustrated", "Hostile"]
        disengagement_intents = ["Hostile disengagement", "Requesting to end chat"]

        # --- End chat if hostile disengagement ---
        if analysis_result.get("user_intent") in disengagement_intents:
            self.conversation_ended = True
            response = strategy_for_persona.get(
                "disengagement_protocol", 
                "I understand you're stressed. Take care, and feel free to reach out when you're ready to discuss options."
            )
            self.memory.chat_memory.add_ai_message(response)
            
            # Update final metrics
            final_score = self._calculate_conversation_success(analysis_result)
            analysis_result['last_turn_score'] = self.success_metrics['last_turn_score']
            analysis_result['cumulative_score'] = final_score
            analysis_result['applied_strategy'] = analysis_result.get('strategy', 'Closure Specialist')
            analysis_result['strike_count'] = f"{self.negative_sentiment_strikes}/{self.max_negative_strikes}"
            
            return response, analysis_result

        # --- Track negative sentiment strikes ---
        if analysis_result.get("sentiment") in negative_sentiments:
            self.negative_sentiment_strikes += 1
            print(f"[STRIKE SYSTEM] Strike {self.negative_sentiment_strikes}/{self.max_negative_strikes} for sentiment: {analysis_result.get('sentiment')}")
            
            if self.negative_sentiment_strikes >= self.max_negative_strikes:
                self.conversation_ended = True
                response = strategy_for_persona.get(
                    "disengagement_protocol",
                    "I understand you're frustrated. To avoid further stress, I'll end our chat now. Please reach out when you're ready to explore solutions."
                )
                self.memory.chat_memory.add_ai_message(response)
                
                # Update final metrics
                final_score = self._calculate_conversation_success(analysis_result)
                analysis_result['last_turn_score'] = self.success_metrics['last_turn_score']
                analysis_result['cumulative_score'] = final_score
                analysis_result['applied_strategy'] = analysis_result.get('strategy', 'Closure Specialist')
                analysis_result['strike_count'] = f"{self.negative_sentiment_strikes}/{self.max_negative_strikes}"
                
                return response, analysis_result

        # --- Conversation Success Score (running average) ---
        cumulative_score = self._calculate_conversation_success(analysis_result)

        # --- Store the real-time LLM-decided strategy ---
        self.last_strategy_used = analysis_result.get('strategy', strategy_for_persona.get("name", self.persona))

        # --- Store metrics in analysis result for sidebar ---
        analysis_result['last_turn_score'] = self.success_metrics['last_turn_score']
        analysis_result['cumulative_score'] = cumulative_score
        analysis_result['applied_strategy'] = self.last_strategy_used
        analysis_result['strike_count'] = f"{self.negative_sentiment_strikes}/{self.max_negative_strikes}"

        # --- RAG Context ---
        rag_context = self._get_rag_context(customer_utterance)

        # --- Core Prompt Template (Updated to use real-time strategy) ---
        generation_prompt_template = """
        You are an expert collections assistant. Your goal is a natural, empathetic, and compliant conversation to guide the user towards a resolution.

        **1. BASE PERSONA STRATEGY (Your default instruction manual)**
        ---
        {persona_strategy_document}
        ---
        
        **2. REAL-TIME DYNAMIC STRATEGY (LLM-decided strategy for this specific turn)**
        Current Strategy: {real_time_strategy}
        - If "Empathetic Listener": Focus on acknowledging their stress/emotions, offer support
        - If "Solution Provider": Present practical options and alternatives
        - If "Firm but Fair": Be direct about consequences while offering reasonable solutions
        - If "Information Clarifier": Answer their questions clearly and provide specific details
        - If "Negotiation Facilitator": Work with them to find mutually acceptable terms
        - If "Closure Specialist": Wrap up respectfully, provide next steps
        ---
        
        **3. CUSTOMER PROFILE & FINANCIALS (All amounts are in Indian Rupees - INR)**
        ---
        {financial_details}
        ---
        
        **4. PREDICTIVE MODEL INSIGHTS (Default probability: {default_prob:.1%})**
        ---
        {prediction_features}
        ---
        
        **5. PERSONA MODEL INSIGHTS (Classified as: {persona})**
        ---
        {persona_features}
        ---
        
        **6. REAL-TIME CONVERSATION ANALYSIS:**
        - Latest Customer Sentiment: {sentiment}
        - User's Inferred Intent: {user_intent}
        - Strike Count: {strike_count}
        - Turn Score: {last_turn_score}/100
        - Cumulative Score: {cumulative_score:.1f}/100
        - Offers Already Made/Rejected: {offers_made}
        - Full Conversation History: {history}
        ---
        
        **7. RELEVANT COMPLIANCE GUIDELINES (from knowledge base)**
        ---
        {rag_context}
        ---
        
        **YOUR TASK & STRICT RULES:**
        Write the next response following the REAL-TIME DYNAMIC STRATEGY while staying compliant:
        1. **PRIMARY STRATEGY**: Follow the "{real_time_strategy}" approach for this turn
        2. **ACKNOWLEDGE AND ADVANCE**: Address their last message based on sentiment/intent, then guide toward resolution
        3. **USE ALL CONTEXT**: Let guidelines, predictive insights, and persona insights influence your response
        4. **NEVER REPEAT OFFERS**: Avoid suggesting amounts already made/rejected: {offers_made}
        5. **BE ACTION-ORIENTED**: Present 1-2 relevant options when appropriate
        6. **BE CONCISE**: 1-2 sentences max (≤50 words), maintain empathetic but professional tone

        Bot Response:
        """
        
        prompt = PromptTemplate(
            template=generation_prompt_template,
            input_variables=[
                "persona_strategy_document", "real_time_strategy", "financial_details", 
                "default_prob", "prediction_features", "persona", "persona_features",
                "sentiment", "user_intent", "strike_count", "last_turn_score", 
                "cumulative_score", "offers_made", "history", "rag_context"
            ]
        )

        pred_features_rounded = {k: f"{v:.2f}" for k, v in self.prediction_features.items() if isinstance(v, (int, float))}
        pers_features_rounded = {k: f"{v:.2f}" for k, v in self.persona_features.items() if isinstance(v, (int, float))}

        prompt_filled = prompt.format(
            persona_strategy_document=json.dumps(strategy_for_persona, indent=2),
            real_time_strategy=self.last_strategy_used,
            financial_details=json.dumps(self.financial_details, indent=2),
            default_prob=self.default_prob,
            prediction_features=json.dumps(pred_features_rounded, indent=2),
            persona=self.persona,
            persona_features=json.dumps(pers_features_rounded, indent=2),
            sentiment=analysis_result['sentiment'],
            user_intent=analysis_result['user_intent'],
            strike_count=f"{self.negative_sentiment_strikes}/{self.max_negative_strikes}",
            last_turn_score=self.success_metrics['last_turn_score'],
            cumulative_score=cumulative_score,
            offers_made=list(self.offers_made) if self.offers_made else "None",
            history=history,
            rag_context=rag_context
        )

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_filled}],
                model=self.groq_model_name,
                temperature=0.5
            )
            bot_response = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"[LLM GENERATION ERROR] Failed: {e}")
            bot_response = "I'm having a technical issue at the moment. Please give me a moment."

        self._track_offers(bot_response)
        self.memory.chat_memory.add_ai_message(bot_response)
        return bot_response, analysis_result
