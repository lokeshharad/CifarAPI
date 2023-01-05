import logging

from datetime import datetime
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

time_now = time.time()
date = datetime.fromtimestamp(time_now).strftime("%Y-%m-%d")

formatter = logging.Formatter("%(asctime)s %(levelname)s  %(name)s  : %(message)s", datefmt=' %Y/%m/%d  %H:%M:%S ')
handler = logging.FileHandler("logs/%s.log" %(date), "a")
handler.setFormatter(formatter)
logger.addHandler(handler)
