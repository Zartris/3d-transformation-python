import airsim

from Data.UE4Data import convert_airsim_data

if __name__ == '__main__':
    # ================================================================================================================
    # Connect to UE4 Server
    # ================================================================================================================
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    pose = convert_airsim_data(client.simGetObjectPose("PX4"))
    print(pose)
    print(client.getClientVersion())
    print(client.getServerVersion())

    # ================================================================================================================
    # Extract UE4 Data
    # ================================================================================================================
