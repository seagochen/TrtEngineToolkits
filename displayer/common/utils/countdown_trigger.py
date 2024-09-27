class CountdownTrigger:

    def __init__(self, trigger_count=10):
        self.countdown_trigger = 0
        self.trigger_count = trigger_count

    def perform(self):

        if self.countdown_trigger == 0:
            # リセットtrigger
            self.countdown_trigger = self.trigger_count
            return True

        # カウントダウン
        self.countdown_trigger -= 1
        return False
