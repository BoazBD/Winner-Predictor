class Bet:
    def __init__(self, bet_type, event_date, time, event, option1, ratio1, option2, ratio2, option3=None, ratio3=None):
        self.bet_type = bet_type
        self.event_date = event_date
        self.time = time
        self.event = event
        self.option1 = option1
        self.ratio1 = ratio1
        self.option2 = option2
        self.ratio2 = ratio2
        self.option3 = option3
        self.ratio3 = ratio3
    
    def __str__(self):
        """
        Return a string representation of the Bet object.
        """
        result = f"Bet Type: {self.bet_type}, Date: {self.event_date}, Time: {self.time}, Event: {self.event}, Option1: {self.option1}, Ratio1: {self.ratio1}, Option2: {self.option2}, Ratio2: {self.ratio2}"
        if self.option3 and self.ratio3:
            result += f", Option3: {self.option3}, Ratio3: {self.ratio3}"
        return result
