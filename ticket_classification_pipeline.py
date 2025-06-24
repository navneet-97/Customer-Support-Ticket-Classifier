import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
import pickle
import json
from typing import Dict, List, Tuple, Any
from typing import Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Sentiment analysis
from textblob import TextBlob

# Gradio for web interface
import gradio as gr

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

class TicketClassificationPipeline:
    """
    Complete ML pipeline for customer support ticket classification and entity extraction.
    """
    
    def __init__(self):
        self.issue_type_model = None
        self.urgency_model = None
        self.tfidf_vectorizer = None
        self.issue_type_encoder = None
        self.urgency_encoder = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Product list for entity extraction
        self.products = [
            "SmartWatch V2", "UltraClean Vacuum", "SoundWave 300", "PhotoSnap Cam",
            "Vision LED TV", "EcoBreeze AC", "RoboChef Blender", "FitRun Treadmill",
            "PowerMax Battery", "ProTab X1"
        ]
        
        # Complaint keywords
        self.complaint_keywords = [
            'broken', 'defective', 'damaged', 'faulty', 'error', 'issue', 'problem',
            'late', 'delayed', 'wrong', 'incorrect', 'missing', 'bad', 'poor',
            'not working', 'failed', 'crash', 'bug', 'slow', 'stuck'
        ]
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from Excel file.
        """
        try:
            df = pd.read_excel(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\#]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize text and apply lemmatization.
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token.lower() not in self.stop_words and token.isalpha():
                lemmatized = self.lemmatizer.lemmatize(token.lower())
                processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features from text data.
        """
        feature_df = df.copy()
        
        # Text length features
        feature_df['text_length'] = feature_df['ticket_text'].astype(str).apply(len)
        feature_df['word_count'] = feature_df['ticket_text'].astype(str).apply(lambda x: len(x.split()))
        
        # Sentiment features
        feature_df['sentiment_polarity'] = feature_df['ticket_text'].astype(str).apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        feature_df['sentiment_subjectivity'] = feature_df['ticket_text'].astype(str).apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )
        
        # Urgency indicators
        urgency_words = ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'now']
        feature_df['has_urgency_words'] = feature_df['ticket_text'].astype(str).apply(
            lambda x: int(any(word in x.lower() for word in urgency_words))
        )
        
        # Question indicators
        feature_df['has_question'] = feature_df['ticket_text'].astype(str).apply(
            lambda x: int('?' in x)
        )
        
        # Complaint indicators
        feature_df['complaint_score'] = feature_df['ticket_text'].astype(str).apply(
            lambda x: sum(1 for word in self.complaint_keywords if word in x.lower())
        )
        
        return feature_df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete data preprocessing pipeline.
        """
        print("Starting data preprocessing...")
        
        # Clean text
        df['cleaned_text'] = df['ticket_text'].apply(self.clean_text)
        
        # Extract additional features
        df = self.extract_features(df)
        
        # Handle missing values
        if 'issue_type' in df.columns:
            df['issue_type'] = df['issue_type'].fillna('Unknown')
        if 'urgency_level' in df.columns:
            df['urgency_level'] = df['urgency_level'].fillna('Low')

        # Remove rows with empty text
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"Preprocessing completed. Final shape: {df.shape}")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for machine learning models.
        """
        # TF-IDF features
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(df['cleaned_text'])
        else:
            tfidf_features = self.tfidf_vectorizer.transform(df['cleaned_text'])

        # Additional numerical features
        numerical_features = df[[
            'text_length', 'word_count', 'sentiment_polarity', 
            'sentiment_subjectivity', 'has_urgency_words', 
            'has_question', 'complaint_score'
        ]].values

        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([tfidf_features, numerical_features])

        # Safely return labels only if available
        if 'issue_type' in df.columns and 'urgency_level' in df.columns:
            return combined_features, df[['issue_type', 'urgency_level']].values
        else:
            return combined_features, None

    def train_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train both issue type and urgency level classification models.
        """
        print("Training models...")
        
        # Prepare labels
        issue_types = y[:, 0]
        urgency_levels = y[:, 1]
        
        # Encode labels
        self.issue_type_encoder = LabelEncoder()
        self.urgency_encoder = LabelEncoder()
        
        issue_types_encoded = self.issue_type_encoder.fit_transform(issue_types)
        urgency_levels_encoded = self.urgency_encoder.fit_transform(urgency_levels)
        
        # Split data
        X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test = train_test_split(
            X, issue_types_encoded, urgency_levels_encoded, test_size=0.2, random_state=42, stratify=issue_types_encoded
        )
        
        # Train Issue Type Classifier
        print("Training Issue Type Classifier...")
        issue_type_models = {
            'logistic': LogisticRegression(max_iter=2000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        best_issue_score = 0
        for name, model in issue_type_models.items():
            model.fit(X_train, y_issue_train)
            score = model.score(X_test, y_issue_test)
            print(f"{name} Issue Type Accuracy: {score:.4f}")
            
            if score > best_issue_score:
                best_issue_score = score
                self.issue_type_model = model
        
        # Train Urgency Level Classifier
        print("Training Urgency Level Classifier...")
        urgency_models = {
            'logistic': LogisticRegression(max_iter=2000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        best_urgency_score = 0
        for name, model in urgency_models.items():
            model.fit(X_train, y_urgency_train)
            score = model.score(X_test, y_urgency_test)
            print(f"{name} Urgency Level Accuracy: {score:.4f}")
            
            if score > best_urgency_score:
                best_urgency_score = score
                self.urgency_model = model
        
        # Evaluate models
        self.evaluate_models(X_test, y_issue_test, y_urgency_test)
    
    def evaluate_models(self, X_test: np.ndarray, y_issue_test: np.ndarray, y_urgency_test: np.ndarray) -> None:
        """
        Evaluate trained models and print detailed metrics.
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Issue Type Classification Results
        issue_pred = self.issue_type_model.predict(X_test)
        print("\nISSUE TYPE CLASSIFICATION:")
        print("-" * 30)
        print("Accuracy:", accuracy_score(y_issue_test, issue_pred))
        print("\nClassification Report:")
        print(classification_report(y_issue_test, issue_pred, 
                                    target_names=self.issue_type_encoder.classes_))
        
        # Urgency Level Classification Results
        urgency_pred = self.urgency_model.predict(X_test)
        print("\nURGENCY LEVEL CLASSIFICATION:")
        print("-" * 30)
        print("Accuracy:", accuracy_score(y_urgency_test, urgency_pred))
        print("\nClassification Report:")
        print(classification_report(y_urgency_test, urgency_pred, 
                                    target_names=self.urgency_encoder.classes_,zero_division=0))
    
    def extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text using regex patterns.
        """
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD Month YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',  # Month DD, YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return dates
    
    def extract_products(self, text: str) -> List[str]:
        """
        Extract product names from text.
        """
        found_products = []
        text_lower = text.lower()
        
        for product in self.products:
            if product.lower() in text_lower:
                found_products.append(product)
        
        return found_products
    
    def extract_complaint_keywords(self, text: str) -> List[str]:
        """
        Extract complaint keywords from text.
        """
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in self.complaint_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def extract_order_numbers(self, text: str) -> List[str]:
        """
        Extract order numbers from text.
        """
        # Pattern for order numbers like #12345, order #12345, etc.
        pattern = r'#\d{4,6}|order\s*#?\s*\d{4,6}'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return matches
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all entities from ticket text.
        """
        entities = {
            'products': self.extract_products(text),
            'dates': self.extract_dates(text),
            'complaint_keywords': self.extract_complaint_keywords(text),
            'order_numbers': self.extract_order_numbers(text)
        }
        
        return entities
    
    def predict_ticket(self, ticket_text: str) -> Dict[str, Any]:
        """
        Main prediction function that combines classification and entity extraction.
        """
        if not self.issue_type_model or not self.urgency_model:
            raise ValueError("Models not trained. Please train the models first.")
        
        # Create a temporary dataframe for preprocessing
        temp_df = pd.DataFrame({'ticket_text': [ticket_text]})
        temp_df = self.preprocess_data(temp_df)
        
        # Prepare features
        X, _ = self.prepare_features(temp_df)
        
        # Make predictions
        issue_type_pred = self.issue_type_model.predict(X)[0]
        urgency_pred = self.urgency_model.predict(X)[0]
        
        # Get prediction probabilities
        issue_type_proba = self.issue_type_model.predict_proba(X)[0]
        urgency_proba = self.urgency_model.predict_proba(X)[0]
        
        # Decode predictions
        predicted_issue_type = self.issue_type_encoder.inverse_transform([issue_type_pred])[0]
        predicted_urgency = self.urgency_encoder.inverse_transform([urgency_pred])[0]
        
        # Extract entities
        entities = self.extract_entities(ticket_text)
        
        # Prepare confidence scores
        issue_confidence = dict(zip(self.issue_type_encoder.classes_, issue_type_proba))
        urgency_confidence = dict(zip(self.urgency_encoder.classes_, urgency_proba))
        
        result = {
            'predicted_issue_type': predicted_issue_type,
            'predicted_urgency_level': predicted_urgency,
            'extracted_entities': entities,
            'confidence_scores': {
                'issue_type': issue_confidence,
                'urgency_level': urgency_confidence
            }
        }
        
        return result
    
    def save_models(self, filepath: str) -> None:
        """
        Save trained models and preprocessors.
        """
        model_data = {
            'issue_type_model': self.issue_type_model,
            'urgency_model': self.urgency_model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'issue_type_encoder': self.issue_type_encoder,
            'urgency_encoder': self.urgency_encoder,
            'products': self.products,
            'complaint_keywords': self.complaint_keywords
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """
        Load trained models and preprocessors.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.issue_type_model = model_data['issue_type_model']
        self.urgency_model = model_data['urgency_model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.issue_type_encoder = model_data['issue_type_encoder']
        self.urgency_encoder = model_data['urgency_encoder']
        self.products = model_data['products']
        self.complaint_keywords = model_data['complaint_keywords']
        
        print(f"Models loaded from {filepath}")

def create_gradio_interface(pipeline: TicketClassificationPipeline) -> gr.Interface:
    """
    Create Gradio web interface for the ticket classification system.
    """
    def process_ticket(ticket_text):
        if not ticket_text.strip():
            return "Please enter a ticket description.", "", "{}"
        
        try:
            result = pipeline.predict_ticket(ticket_text)
            
            # Format the output
            issue_type = result['predicted_issue_type']
            urgency = result['predicted_urgency_level']
            entities = result['extracted_entities']
            confidence = result['confidence_scores']
            
            # Create formatted output
            prediction_text = f"Issue Type: {issue_type}\nUrgency Level: {urgency}"
            
            # Format confidence scores
            confidence_text = "Confidence Scores:\n"
            confidence_text += f"Issue Type: {confidence['issue_type'][issue_type]:.3f}\n"
            confidence_text += f"Urgency Level: {confidence['urgency_level'][urgency]:.3f}"
            
            # Format entities as JSON
            entities_json = json.dumps(entities, indent=2)
            
            return prediction_text, confidence_text, entities_json
            
        except Exception as e:
            return f"Error: {str(e)}", "", "{}"
    
    # Create the interface
    interface = gr.Interface(
        fn=process_ticket,
        inputs=gr.Textbox(
            lines=5,
            placeholder="Enter customer support ticket text here...",
            label="Ticket Text"
        ),
        outputs=[
            gr.Textbox(label="Predictions", lines=3),
            gr.Textbox(label="Confidence Scores", lines=3),
            gr.Textbox(label="Extracted Entities (JSON)", lines=10)
        ],
        title="Customer Support Ticket Classifier",
        description="Enter a customer support ticket to get automatic classification and entity extraction.",
        examples=[
            ["My SmartWatch V2 is broken and won't turn on. I need urgent help!"],
            ["Can you tell me about the warranty for UltraClean Vacuum?"],
            ["I ordered SoundWave 300 on 15 March but received the wrong item. Order #12345."],
            ["PhotoSnap Cam installation failed at step 3. Getting error message."]
        ]
    )
    
    return interface

def main():
    """
    Main execution function.
    """
    print("="*60)
    print("CUSTOMER SUPPORT TICKET CLASSIFICATION PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = TicketClassificationPipeline()
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = pipeline.load_data('ai_dev_assignment_tickets_complex_1000.xls')
    if df is None:
        return
    
    df_processed = pipeline.preprocess_data(df)
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y = pipeline.prepare_features(df_processed)
    
    # Train models
    print("\n3. Training models...")
    pipeline.train_models(X, y)
    
    # Save models
    print("\n4. Saving models...")
    pipeline.save_models('ticket_classification_models.pkl')
    
    # Test the prediction function
    print("\n5. Testing prediction function...")
    sample_tickets = [
        "My SmartWatch V2 is broken and won't turn on. I need urgent help!",
        "Can you tell me about the warranty for UltraClean Vacuum?",
        "I ordered SoundWave 300 on 15 March but received wrong item. Order #12345."
    ]
    
    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\nSample {i}:")
        print(f"Input: {ticket}")
        result = pipeline.predict_ticket(ticket)
        print(f"Predicted Issue Type: {result['predicted_issue_type']}")
        print(f"Predicted Urgency: {result['predicted_urgency_level']}")
        print(f"Extracted Entities: {result['extracted_entities']}")
    
    # Create and launch Gradio interface
    print("\n6. Creating Gradio interface...")
    interface = create_gradio_interface(pipeline)
    
    print("\nPipeline setup complete!")
    print("To launch the web interface, run: interface.launch()")
    
    return pipeline, interface

if __name__ == "__main__":
    pipeline, interface = main()
    
    interface.launch(share=True)