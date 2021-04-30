import sys
import os

# Source: https://cx-freeze.readthedocs.io/en/latest/faq.html#using-data-files
def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)
    return os.path.join(datadir, filename)

# Konstanten für die Darstellung des Icons
TRAY_TOOLTIP = find_data_file('Voice Assistant')
TRAY_ICON_INITIALIZING = find_data_file('initializing.png')
TRAY_ICON_IDLE = find_data_file('idle.png')
TRAY_ICON_LISTENING = find_data_file('listening.png')
TRAY_ICON_SPEAKING = find_data_file('speaking.png')