import paramiko
import scp
import getpass
import bz2
import os.path as os_path
import os

REMOTE_PATH = '/home/bcstri01/env2/logdir/'
LOCAL_PATH = '/home/redbird_general/Desktop/CRC/'
port = 22
server = 'crc.hpc.louisville.edu'
user = 'bcstri01'

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

password = getpass.getpass(user + "'s password:")
# if os_path.isfile("passwdFile"):
#     passwdFile = open('passwdFile', "r")
#     password = bz2.decompress(passwdFile.read("passwdFile"))
# else:
#     passwdFile = open("passwdFile", "w")
#     password = getpass.getpass(user + "'s password:")
#     passwdFile.writelines(bz2.compress(password.encode('utf-8')))
# passwdFile.close()
ssh = createSSHClient(server, port, user, password)

print('SSH client created')

scp = scp.SCPClient(ssh.get_transport())
print("Downloading CRC's " + REMOTE_PATH + "to local machine's " + LOCAL_PATH )
scp.get(remote_path=REMOTE_PATH, local_path=LOCAL_PATH, recursive=True)
print("download complete")