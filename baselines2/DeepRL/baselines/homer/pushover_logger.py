# from pushover import Client


# Add the client and API token to send messages to your phone.
CLIENT_TOKEN = ""
API_TOKEN = ""


class PushoverLogger(object):
    def __init__(self, experiment_name):
        return
        self.experiment_name = experiment_name
        andrew_client = Client(CLIENT_TOKEN, api_token=API_TOKEN)
        self.clients = [andrew_client]

    def log(self, message):
        return
        for client in self.clients:
            client.send_message(message, title=self.experiment_name)

