from client_app import BenignClient
from attacks.fake_label_distribution import fake_label_distribution

class AttackClient(BenignClient):
    def _get_label_distribution(self, trainloader):
        return fake_label_distribution(trainloader)