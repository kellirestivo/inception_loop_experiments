def get_filename():
    """
    Prompts user for root filename of current experiment.

    Args:
        None
    Returns:
        str: The root filename input by the user.
    """

    return input("Input root filename = ")

def get_dotmap_position(monitor_x=1920, monitor_y=1080):
    """
    Prompts the user for x and y coordinates and returns them as a tuple.

    Args:
        monitor_x (int): number of pixels on horizontal axis of monitor
        monitor_y (int): number of pixels on vertical axis of monitor
    Returns:
        tuple: (user_input_x, user_input_y) coordinates in monitor space (1920x1080).
    """
    while True:
        try:
            user_input_x = int(input("Please enter the x coordinate (in monitor space 1920x1080): "))
            user_input_y = int(input("Please enter the y coordinate (in monitor space 1920x1080): "))
            
            if 0 <= user_input_x <= monitor_x and 0 <= user_input_y <= monitor_y:
                break
            else:
                print("Coordinates must be within the range (0, 0) to (1920, 1080). Please try again.")
        except ValueError:
            print("Invalid input. Please enter integer values for the coordinates.")

    dotmap_position = (user_input_x, user_input_y)
    print(f"Dotmap position: {dotmap_position}")

    return dotmap_position