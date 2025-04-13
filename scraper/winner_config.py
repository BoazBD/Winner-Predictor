import random
import uuid

RECORDING_BETS = [
    "הימור יתרון - תוצאת סיום (ללא הארכות)",  # Soccer
    "‮1X2‬ - תוצאת סיום (ללא הארכות)",  # Soccer
    "הימור יתרון - ללא הארכות",  # Basketball
    "הימור יתרון - כולל הארכות אם יהיו",  # Football
    "מחצית/סיום - ללא הארכות",  # Football
    "המנצח/ת - משחק",  # Tennis
]
SID_MAP = {
    240: "Soccer",
    227: "Basketball",
    1100: "Handball",
    1: "Football",
    239: "Tennis",
    226: "Baseball",
}
HASH_CHECKSUM_URL = "https://api.winner.co.il/v2/publicapi/GetCMobileHashes"
API_URL = "https://api.winner.co.il/v2/publicapi/GetCMobileLine"
RESULTS_URL = "https://www.winner.co.il/api/v2/publicapi/GetResults"


headers = {
    "Deviceid": f"2e7f{random.randint(0,9)}66a5ff1{random.randint(0,9)}9d4a122e3d{random.randint(0,9)}5b35a0{random.randint(0,9)}",
    "Hashesmessage": '{"reason":"Initiated"}',
    "Host": "api.winner.co.il",
    "Origin": "https://www.winner.co.il",
    "Referer": "https://www.winner.co.il/",
    "Requestid": str(uuid.uuid4()).replace("-", ""),
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Useragentdatresa": '{"devicemodel":"Macintosh","deviceos":"mac os","deviceosversion":"10.15.7","appversion":"1.8.2","apptype":"mobileweb","originId":"3","isAccessibility":false}',
    "X-Csrf-Token": "null",
}
