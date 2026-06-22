from datetime import datetime

def get_current_time(timezone: str = "IST") -> dict:
    """Get current time"""
    now = datetime.now()
    return {
        "timezone": timezone,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "day": now.strftime("%A")
    }