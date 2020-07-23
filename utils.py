import os
import socket
import warnings
import matplotlib
import slack

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def _load_slack_token() -> str:
    with open('~/slacktoken.txt', 'r') as f:
        token = f.read()
    return token


def save_plot(filepath: str, to_slack: bool = False, **kwargs) -> None:
    """
	Function to parse the plt.savefig method and send the file to a slack channel.
	:param filepath: The path+filename+extension where to save the figure
	:param to_slack: Gives the option to send the the plot to slack
	:param kwargs: Other kwargs to parse into the plt.savefig
	:return: None
	"""
    plt.savefig(filepath, **kwargs)
    if to_slack:
        print(
            "[+] Forwarding to the `#personal` Slack channel",
            f"\tDir: {os.path.dirname(filepath)}",
            f"\tFile: {os.path.basename(filepath)}",
            sep='\n'
        )
        try:
            slack_token = _load_slack_token()
            slack_msg = f"Host: {socket.gethostname()}\nDir: {os.path.dirname(filepath)}\nFile: {os.path.basename(filepath)}"
            # Send files to Slack: init slack client with access token
            client = slack.WebClient(token=slack_token)
            client.files_upload(file=filepath, initial_comment=slack_msg, channels='#personal')
        except:
            warnings.warn("[-] Failed to broadcast plot to Slack channel.")
