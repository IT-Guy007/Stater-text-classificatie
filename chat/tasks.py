from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
#from chatterbot import ChatBot
#from chatterbot.ext.django_chatterbot import settings
import requests
import time

channel_layer = get_channel_layer()


@shared_task
def get_response(channel_name, input_data):
    #chatterbot = ChatBot(**settings.CHATTERBOT)
    #response = chatterbot.get_response(input_data)
    #response_data = response.serialize()

    external_api_url = "http://127.0.0.1:5000/reply?question=" + input_data
    # res = urllib.request.urlopen(external_api_url).read()
    # category = json.loads(res)
    # category = res.load()
    category = requests.get(external_api_url, headers=headers, verify=False)
    time.sleep(5)
    async_to_sync(channel_layer.send)(
        channel_name,
        {
            "type": "chat.message",
            "text": {"msg": category.text, "source": "bot"},
        },
    )
