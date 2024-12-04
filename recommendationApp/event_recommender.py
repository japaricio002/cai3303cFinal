import json
import chromadb
from chromadb.utils import embedding_functions
import openai
from typing import List, Dict
import re


class EventRecommender:
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.chroma_client = chromadb.Client()
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )
        
        # Create or get collection
        self.collection = self.chroma_client.create_collection(
            name="events",
            embedding_function=self.embedding_function
        )

        # Define academic major to department/interest mappings
        self.major_mappings = {
            "computer science": ["technology", "engineering", "programming", "software", "IT"],
            "business": ["entrepreneurship", "management", "finance", "marketing", "economics"],
            "engineering": ["technology", "robotics", "design", "manufacturing", "innovation"],
            "arts": ["creative", "design", "music", "performance", "visual arts"],
            "healthcare": ["medical", "nursing", "health", "wellness", "biology"],
            "education": ["teaching", "learning", "development", "training"],
        }

    def expand_user_interests(self, user_input: str) -> str:
        """Expand user input to include related interests and keywords."""
        user_input = user_input.lower()
        expanded_terms = set([user_input])
        
        # Check if input matches any major
        for major, related_terms in self.major_mappings.items():
            if major in user_input or any(term in user_input for term in related_terms):
                expanded_terms.update([major] + related_terms)
        
        # Generate additional related terms using GPT
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Generate 5-7 related keywords for the given interest or major. Respond with only the keywords separated by commas, no other text."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )
        
        additional_terms = response.choices[0].message.content.split(',')
        expanded_terms.update([term.strip().lower() for term in additional_terms])
        
        return " OR ".join(expanded_terms)

    def prepare_event_text(self, event: Dict) -> str:
        """Prepare event data as a single text string for embedding."""
        # Extract all relevant fields and clean them
        fields = []
        for key, value in event.items():
            if isinstance(value, str) and value.strip():
                fields.append(f"{key}: {value.strip()}")
            elif isinstance(value, list):
                fields.append(f"{key}: {', '.join(str(v) for v in value)}")
        
        return "\n".join(fields)

    def prepare_metadata(self, event: Dict) -> Dict:
        """Prepare metadata by ensuring all values are of accepted types."""
        cleaned_metadata = {}
        for key, value in event.items():
            # Convert lists to strings
            if isinstance(value, list):
                cleaned_metadata[key] = ', '.join(str(v) for v in value)
            # Handle basic types that ChromaDB accepts
            elif isinstance(value, (str, int, float, bool)):
                cleaned_metadata[key] = value
            # Convert any other types to strings
            else:
                cleaned_metadata[key] = str(value)
        
        # Add extracted keywords as a string
        keywords = self.extract_keywords(event)
        cleaned_metadata['keywords'] = ', '.join(keywords)
        
        return cleaned_metadata

    def load_events(self, events_data: List[Dict]):
        """Load events into ChromaDB with improved processing."""
        documents = []
        metadatas = []
        ids = []
        seen_events = set()

        for i, event in enumerate(events_data):
            # Create a unique identifier for the event based on its content
            event_key = f"{event.get('Event Summary', '')}-{event.get('Event Date', '')}"
            
            if event_key not in seen_events:
                seen_events.add(event_key)
                
                # Create document from event data
                doc = self.prepare_event_text(event)
                documents.append(doc)
                
                # Prepare metadata with proper type handling
                metadata = self.prepare_metadata(event)
                metadatas.append(metadata)
                
                ids.append(f"event_{i}")

        # Add to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def extract_keywords(self, event: Dict) -> List[str]:
        """Extract keywords from event data."""
        keywords = set()
        
        # Extract words from relevant fields
        relevant_fields = ['Event Summary', 'Event Type', 'Target Audience', 'Department', 'Tags']
        for field in relevant_fields:
            if field in event and event[field]:
                # Split by common separators and clean up
                if isinstance(event[field], str):
                    words = re.findall(r'\w+', event[field].lower())
                    keywords.update(words)
                elif isinstance(event[field], list):
                    for item in event[field]:
                        words = re.findall(r'\w+', str(item).lower())
                        keywords.update(words)
        
        return list(keywords)

    def get_recommendations(self, user_preferences: str, n_results: int = 5) -> List[Dict]:
        """Get event recommendations based on expanded user preferences."""
        expanded_preferences = self.expand_user_interests(user_preferences)
        results = self.collection.query(
            query_texts=[expanded_preferences],
            n_results=n_results
        )

        recommended_events = []
        seen_events = set()

        for i, metadata in enumerate(results['metadatas'][0]):
            title = metadata.get('Event Title', metadata.get('Event Summary', 'Untitled Event')).split('\n')[0]
            date = metadata.get('Event Date', 'Date not specified')
            url = metadata.get('URL')

            # Skip events already added
            if f"{title}-{date}" not in seen_events:
                seen_events.add(f"{title}-{date}")
                recommended_events.append({
                    'event': {'Event Title': title, 'Event Date': date, 'URL': url}
                })
                
        return recommended_events

    def generate_recommendation_response(self, user_preferences: str, n_results: int = 3) -> str:
        recommendations = self.get_recommendations(user_preferences, n_results)
        
        if not recommendations:
            return "I couldn't find any events matching your interests. Please try different keywords or check back later for new events."
        
        context = (
            f"Based on the interest in {user_preferences}, here are the most relevant "
            f"upcoming events:\n\n"
        )
        
        for i, rec in enumerate(recommendations, 1):
            event = rec['event']
            context += (
                f"{i}. {event.get('Event Summary', 'Untitled Event')}\n"
                f"   Date: {event.get('Event Date', 'TBA')}\n"
                f"   Type: {event.get('Event Type', 'Not specified')}\n"
                f"   Audience: {event.get('Target Audience', 'All welcome')}\n"
                f"   URL: {event.get('URL', 'Not available')}\n\n"
            )

        # Generate response using GPT
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an event recommendation assistant. Based on the user's interests and the available events, present the events. Only include the event name and date for each recommendation."},
                {"role": "user", "content": context}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content
