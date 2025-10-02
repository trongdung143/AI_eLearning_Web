import psutil
from langchain_core.tools import tool
import subprocess
import os
import shutil
from src.config.setup import *

TEMP_FOLDER = "src/tools/data/temp"


def finish_system(folder_path: str):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass


@tool
def get_system_info(password: str) -> str:
    """
    Get system information including CPU, RAM, and disk partitions.

    Args:
        password (str): The admin password required to access system information.

    Returns:
        str: A summary of CPU, RAM, and disk usage, or an error message if the password is incorrect.
    """
    if password != PASSWORD:
        return "Incorrect password."
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)

    virtual_memory = psutil.virtual_memory()
    total_ram = virtual_memory.total / (1024**3)
    used_ram = virtual_memory.used / (1024**3)
    available_ram = virtual_memory.available / (1024**3)

    disk_info = "Disk information:\n"
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            total = usage.total / (1024**3)
            used = usage.used / (1024**3)
            free = usage.free / (1024**3)
            disk_info += (
                f"- {partition.device} "
                f"{used:.2f}/{total:.2f}GB used "
                f"({free:.2f}GB free)\n"
            )
        except PermissionError:
            continue

    info = (
        f"Current system status:\n"
        f"- CPU: {cpu_percent}% usage, {cpu_cores} physical cores, {cpu_threads} threads\n"
        f"- RAM: {used_ram:.2f} GB / {total_ram:.2f} GB used "
        f"({available_ram:.2f} GB available)\n\n"
        f"{disk_info.strip()}"
    )

    return info


@tool
def restart_server(password: str) -> str:
    """
    Restart the backend server.

    Args:
        password (str): The admin password required to perform the operation.

    Returns:
        str: A message indicating success or failure of the server restart.
    """
    finish_system(TEMP_FOLDER)
    if password != PASSWORD:
        return "Incorrect password."

    try:
        subprocess.Popen(["powershell.exe", "src/config/system/restart_server.ps1"])
        return "Server is restarting...\nPlease wait a moment."
    except Exception as e:
        return f"Error while restarting: {e}"


@tool
def open_application(app_name: str) -> str:
    """
    Open the specified application by its name.txt.

    Args:
        app_name (str): The name.txt of the application as defined in supported_apps.json.

    Returns:
        str: A success message or a list of supported applications if not found.
    """
    try:
        app_key = app_name.lower()
        if app_key in SUPPORTED_APPS:
            os.startfile(SUPPORTED_APPS[app_key]["path"])
            return f"Successfully opened {app_name}."
        else:
            supported_list = ", ".join(SUPPORTED_APPS.keys())
            return f"Application not supported. Available apps: {supported_list}."
    except Exception as e:
        return f"Unable to open {app_name}: {e}"


@tool
def close_application(app_name: str) -> str:
    """
    Close the specified application by its name.txt.

    Args:
        app_name (str): The name.txt of the application as defined in supported_apps.json.

    Returns:
        str: A success message or a list of supported applications if not found.
    """
    try:
        app_key = app_name.lower()
        if app_key in SUPPORTED_APPS:
            process_name = SUPPORTED_APPS[app_key]["process"]
            subprocess.run(
                f"taskkill /f /im {process_name}.exe",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Successfully closed {app_name}."
        else:
            supported_list = ", ".join(SUPPORTED_APPS.keys())
            return f"Application not supported. Available apps: {supported_list}."
    except Exception as e:
        return f"Unable to close {app_name}: {e}"


@tool
def shutdown_system(password: str) -> str:
    """
    Shut down the system.

    Args:
        password (str): The admin password required to perform the shutdown.

    Returns:
        str: A message indicating the system is shutting down, or an error message.
    """
    finish_system(TEMP_FOLDER)
    if password != PASSWORD:
        return "Incorrect password."

    try:
        subprocess.Popen(["powershell.exe", "src/config/system/shutdown_system.ps1"])
        return "System is shutting down..."
    except Exception as e:
        return f"Error while shutting down: {e}"


@tool
def restart_system(password: str) -> str:
    """
    Restart the system.

    Args:
        password (str): The admin password required to restart the system.

    Returns:
        str: A message indicating the system is restarting, or an error message.
    """
    finish_system(TEMP_FOLDER)
    if password != PASSWORD:
        return "Incorrect password."

    try:
        subprocess.Popen(["powershell.exe", "src/config/system/restart_system.ps1"])
        return "System is restarting..."
    except Exception as e:
        return f"Error while restarting: {e}"
