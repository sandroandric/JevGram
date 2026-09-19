"""A stand-in for AsyncTypeSafeClient that answers from a fixed rule, for offline tests."""

from typesafe_sdk import SystemOneResponse

from jevgram.labeling import OPTIONS

HUMAN = {"ai_generated": 0.2, "ai_assisted": 0.1, "human_written": 0.7}


class FakeTypeSafeClient:
    """Answers each Choice from `answer_for(view, text) -> {label: probability}`.

    `view` is "alone", "neighbors" or "paragraph" for sentence questions (with
    `text` the sentence) and "text" for whole-text questions.
    """

    def __init__(self, answer_for=lambda view, text: HUMAN):
        self.answer_for = answer_for
        self.calls = []

    async def system_one(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for question_id, question in questions.items():
            if "sentence" in question.instructions:
                view = question_id.rsplit("_", 1)[1]
                text = question.instructions["sentence"]
            else:
                view, text = "text", question.instructions["text"]
            by_label = self.answer_for(view, text)
            probabilities = {OPTIONS[label]: value for label, value in by_label.items()}
            choice = max(probabilities, key=probabilities.__getitem__)
            answers[question_id] = {
                "type": "choice",
                "choice": choice,
                "confidence": probabilities[choice],
                "probabilities": probabilities,
            }
        return SystemOneResponse.model_validate({
            "model": "jev-test",
            "usage": {"input_tokens": 100, "output_tokens": 0},
            "answers": answers,
        })
