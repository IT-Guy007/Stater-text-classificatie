import json

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
from .tasks import get_response
import nltk

nltk.download('words')
from googletrans import Translator

from source.models.predictTFIDFmodel import classifiyComplaintTFIDF
from source.models.multinomialLogisticRegression import ask_question
from source.models.randomForest import text_to_prediction_random_forest
from source.models.decisionTree import questionDecisionTree

# from source.models.openaiEmbeddings import question

class ChatConsumer(WebsocketConsumer):
    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        get_response.delay(self.channel_name, text_data_json)

        async_to_sync(self.channel_layer.send)(
            self.channel_name,
            {
                "type": "chat_message",
                "text": {"msg": text_data_json["text"], "source": "user"},
            },
        )

        translation = Translator.translate(text_data_json["text"])

        prediction = "Algorithm not found"
        if "chatgpt" in text_data_json["text"].lower() or 'openai' in text_data_json["text"].lower():
            # classify string using apenai
            pass
        elif "bard" in text_data_json["text"].lower() or 'palm' in text_data_json["text"].lower():
            # classify using bard
            pass
        elif "randomforest" in text_data_json["text"].lower() or 'random forest' in text_data_json["text"].lower():
            # classify using random forest
            prediction = text_to_prediction_random_forest(translation)
            pass
        elif "tfidf" in text_data_json["text"].lower() or 'tf-idf' in text_data_json["text"].lower() or 'tf idf' in text_data_json["text"].lower():
            # classify using random forest
            prediction = classifiyComplaintTFIDF(translation)
            pass
        elif "logistic regression" in text_data_json["text"].lower() or 'logisticregression' in text_data_json["text"].lower():
            # classify using random forest
            prediction = ask_question(translation)
            pass
        elif "decision tree" in text_data_json["text"].lower() or 'decisiontree' in text_data_json["text"].lower():
            # classify using random forest
            prediction = questionDecisionTree(translation)
            pass

        prediction_Translated = Translator.translate("Your question belongs to the: " + prediction + " category",
                                                     dest=translation.src)

        async_to_sync(self.channel_layer.send)(
            self.channel_name,
            {
                "type": "chat_message",
                "text": {"msg": prediction_Translated.text, "source": "bot"},
            },
        )

    def chat_message(self, event):
        text = event["text"]
        self.send(text_data=json.dumps({"text": text}))
