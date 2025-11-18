"""Tests for NLP components."""

import pytest
from src.nlp.intent import IntentClassifier
from src.nlp.entities import EntityExtractor
from src.nlp.sentiment import SentimentAnalyzer
from src.nlp.embeddings import EnhancedEmbeddings


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    def setup_method(self):
        self.classifier = IntentClassifier()

    def test_schedule_intent(self):
        result = self.classifier.classify("Schedule a meeting tomorrow")
        assert result["primary_intent"] == "schedule"
        assert result["confidence"] > 0.3

    def test_question_intent(self):
        result = self.classifier.classify("What time is the meeting?")
        assert result["primary_intent"] == "question"

    def test_reminder_intent(self):
        result = self.classifier.classify("Remind me to call John")
        assert result["primary_intent"] == "reminder"

    def test_greeting_intent(self):
        result = self.classifier.classify("Hello, how are you?")
        assert result["primary_intent"] == "greeting"

    def test_feedback_intent(self):
        result = self.classifier.classify("That's wrong, do this instead")
        assert result["primary_intent"] == "feedback"

    def test_secondary_intents(self):
        result = self.classifier.classify("Can you help me schedule a meeting?")
        assert "secondary_intents" in result
        assert isinstance(result["secondary_intents"], list)

    def test_learning(self):
        self.classifier.learn("user1", "test text", "custom")
        assert self.classifier.user_intent_history["user1"]["custom"] == 1

    def test_custom_intent(self):
        self.classifier.add_custom_intent("pizza", [r"\bpizza\b", r"\border\b"])
        result = self.classifier.classify("I want to order a pizza")
        assert "pizza" in result["all_scores"]


class TestEntityExtractor:
    """Tests for EntityExtractor."""

    def setup_method(self):
        self.extractor = EntityExtractor()

    def test_time_extraction(self):
        result = self.extractor.extract_as_dict("Meeting at 2:30 PM")
        assert len(result["times"]) > 0
        assert result["times"][0]["normalized"] == "14:30"

    def test_date_extraction(self):
        result = self.extractor.extract_as_dict("Let's meet tomorrow")
        assert len(result["dates"]) > 0

    def test_person_extraction(self):
        result = self.extractor.extract_as_dict("Meeting with John Smith")
        assert len(result["people"]) > 0
        assert "John Smith" in result["people"][0]["value"]

    def test_duration_extraction(self):
        result = self.extractor.extract_as_dict("The meeting is for 30 minutes")
        assert len(result["durations"]) > 0
        assert result["durations"][0]["normalized"] == 30

    def test_priority_extraction(self):
        result = self.extractor.extract_as_dict("This is urgent!")
        assert len(result["priorities"]) > 0
        assert result["priorities"][0]["normalized"] == "critical"

    def test_multiple_entities(self):
        result = self.extractor.extract_as_dict(
            "Schedule meeting with John at 3pm tomorrow for 1 hour"
        )
        assert len(result["people"]) > 0
        assert len(result["times"]) > 0
        assert len(result["dates"]) > 0
        assert len(result["durations"]) > 0


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer."""

    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        result = self.analyzer.analyze("This is amazing and wonderful!")
        assert result.polarity > 0.3
        assert result.intensity > 0

    def test_negative_sentiment(self):
        result = self.analyzer.analyze("This is terrible and awful")
        assert result.polarity < -0.3

    def test_neutral_sentiment(self):
        result = self.analyzer.analyze("The meeting is at 3pm")
        assert -0.3 <= result.polarity <= 0.3

    def test_emotion_detection(self):
        result = self.analyzer.analyze("I'm so frustrated with this!")
        assert "frustration" in result.emotions
        assert result.emotions["frustration"] > 0.5

    def test_joy_emotion(self):
        result = self.analyzer.analyze("I'm so happy and excited!")
        assert "joy" in result.emotions

    def test_negation_handling(self):
        result = self.analyzer.analyze("This is not good")
        assert result.polarity < 0

    def test_intensifiers(self):
        weak = self.analyzer.analyze("This is good")
        strong = self.analyzer.analyze("This is extremely good")
        assert strong.intensity >= weak.intensity

    def test_needs_empathy(self):
        assert self.analyzer.needs_empathy("I'm so frustrated and upset")
        assert not self.analyzer.needs_empathy("Schedule a meeting")

    def test_is_frustrated(self):
        assert self.analyzer.is_frustrated("This is so annoying!")
        assert not self.analyzer.is_frustrated("Hello there")


class TestEnhancedEmbeddings:
    """Tests for EnhancedEmbeddings."""

    def setup_method(self):
        self.embeddings = EnhancedEmbeddings()

    def test_embedding_creation(self):
        emb = self.embeddings.create_embedding("Hello world")
        assert emb is not None
        assert len(emb) > 0

    def test_similarity(self):
        sim = self.embeddings.semantic_similarity(
            "Schedule a meeting",
            "Book an appointment"
        )
        assert sim > 0.3

    def test_dissimilarity(self):
        sim = self.embeddings.semantic_similarity(
            "Schedule a meeting",
            "What's for lunch?"
        )
        assert sim < 0.5

    def test_find_similar(self):
        candidates = [
            "Book a meeting room",
            "What's the weather?",
            "Schedule an appointment"
        ]
        results = self.embeddings.find_similar(
            "Schedule a meeting",
            candidates,
            top_k=2
        )
        assert len(results) == 2
        # Most similar should be first
        assert "appointment" in results[0][0].lower() or "meeting" in results[0][0].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
