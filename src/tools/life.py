from langchain_core.tools import tool
from datetime import datetime, timedelta


@tool
def get_time() -> str:
    """
    Get the current local time.

    Returns:
        str: The current time formatted as 'HH:MM:SS DD-MM-YYYY'.
    """
    try:
        now = datetime.now()
        return now.strftime("%H:%M:%S %d-%m-%Y")
    except Exception as e:
        return f"Error while getting time: {str(e)}"


@tool
def get_weather(address: str) -> str:
    """
    Get weather information for a given location.

    Args:
        address (str): The name.txt of the city or address to retrieve weather information for.

    Returns:
        str: Weather information for the specified address, or a default message if unavailable.
    """
    try:
        return f"No weather information available for {address}"
    except Exception as e:
        return f"Error while getting weather: {str(e)}"


@tool
def get_relative_date(days: int) -> str:
    """
    Get the date relative to today by a given number of days.

    Args:
        days (int): Number of days to shift. Negative for past, positive for future.

    Returns:
        str: The calculated date in 'YYYY-MM-DD' format.
    """
    try:
        result_date = datetime.now().date() + timedelta(days=days)
        return result_date.strftime("%Y-%m-%d")
    except Exception as e:
        return f"Error calculating date: {str(e)}"
