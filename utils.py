import os
import warnings
import slack
import socket
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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
        slack_token = 'xoxp-452271173797-451476014913-1101193540773-57eb7b0d416e8764be6849fdeda52ce8'
        slack_msg = f"Host: {socket.gethostname()}\nDir: {os.path.dirname(filepath)}\nFile: {os.path.basename(filepath)}"
        try:
            # Send files to Slack: init slack client with access token
            client = slack.WebClient(token=slack_token)
            client.files_upload(file=filepath, initial_comment=slack_msg, channels='#personal')
        except:
            warnings.warn("[-] Failed to broadcast plot to Slack channel.")
