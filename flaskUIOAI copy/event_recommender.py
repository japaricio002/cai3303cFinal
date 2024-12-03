import json
import chromadb
from chromadb.utils import embedding_functions
import openai
from typing import List, Dict
import os
from datetime import datetime

class EventRecommender:
    def __init__(self, openai_api_key: str):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        
        # Use OpenAI's embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )
        
        # Create or get collection
        self.collection = self.chroma_client.create_collection(
            name="events",
            embedding_function=self.embedding_function
        )

    def prepare_event_text(self, event: Dict) -> str:
        """Prepare event data as a single text string for embedding."""
        return f"""
        Event: {event.get('Event Summary', '')}
        Date: {event.get('Event Date', '')}
        Type: {event.get('Event Type', '')}
        Target Audience: {event.get('Target Audience', '')}
        Department: {event.get('Department', '')}
        Campus: {event.get('Campus', '')}
        Tags: {event.get('Tags', '')}
        """.strip()

    def load_events(self, events_data: List[Dict]):
        """Load events into ChromaDB."""
        documents = []
        metadatas = []
        ids = []

        for i, event in enumerate(events_data):
            # Create document from event data
            doc = self.prepare_event_text(event)
            documents.append(doc)
            
            # Store original event data as metadata
            metadatas.append(event)
            
            # Create unique ID for each event
            ids.append(f"event_{i}")

        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def get_recommendations(self, user_preferences: str, n_results: int = 3) -> List[Dict]:
        """Get event recommendations based on user preferences."""
        # Query the vector store
        results = self.collection.query(
            query_texts=[user_preferences],
            n_results=n_results
        )

        # Get the matched events
        recommended_events = []
        for i in range(len(results['ids'][0])):
            event_metadata = results['metadatas'][0][i]
            similarity = results['distances'][0][i] if 'distances' in results else None
            
            recommended_events.append({
                'event': event_metadata,
                'similarity_score': similarity
            })

        return recommended_events

    def generate_recommendation_response(self, user_preferences: str, n_results: int = 3) -> str:
        """Generate a natural language response with recommendations using GPT."""
        # Get recommendations
        recommendations = self.get_recommendations(user_preferences, n_results)
        
        # Prepare context for GPT
        context = f"User preferences: {user_preferences}\n\nAvailable events:\n"
        for rec in recommendations:
            context += f"\n- {rec['event']['Event Summary']}"
            if 'Event Date' in rec['event']:
                context += f"\n  Date: {rec['event']['Event Date']}"
            if 'Event Type' in rec['event']:
                context += f"\n  Type: {rec['event']['Event Type']}"
            if 'Target Audience' in rec['event']:
                context += f"\n  Target Audience: {rec['event']['Target Audience']}"

        # Generate response using GPT
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an event recommendation assistant. Based on the user's preferences and the available events, suggest the top 3 most relevant events. Only give name of event and date"},
                {"role": "user", "content": context}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = EventRecommender(openai_api_key="")
    
    # Load events from JSON file
    with open("mdc_events.json", "r") as f:
        events_data = json.load(f)
    
    # Load events into the vector store
    recommender.load_events(events_data)
    
    # Example user preferences
    user_preferences = "I'm interested in veteran events."
    
    # Get recommendations with GPT-generated response
    response = recommender.generate_recommendation_response(user_preferences)
    print(response)