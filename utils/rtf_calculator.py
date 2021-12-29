import time

class RTF_calculator():
    def __init__(self, js, fps=100):
        self.js = js
        self.fps = fps
        self.time_stamp = None
    
    def tik(self):
        self.time_stamp = time.time()

    def tok(self):
        time_elapsed = time.time() - self.time_stamp
        time_utts = sum(
                    v["input"][0]["shape"][0] for v in self.js.values()
                    )
        time_utts /= self.fps 

        rtf =  time_elapsed / time_utts
        print("RTF calculator: RTF is {:.2f} | time_utts: {:2f} | time_elapsed: {:.2f}".format(rtf, time_utts, time_elapsed))
