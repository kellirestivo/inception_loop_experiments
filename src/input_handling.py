def get_filename():
    """
    Prompts user for root filename of current experiment.

    Returns:
    str: The root filename input by the user.
    """

    return input("Input root filename = ")

def get_dotmap_position():
    """
    Prompts the user for x and y coordinates and returns them as a tuple.

    Returns:
    tuple: (user_input_x, user_input_y) coordinates in monitor space (1920x1080).
    """
    while True:
        try:
            user_input_x = int(input("Please enter the x coordinate (in monitor space 1920x1080): "))
            user_input_y = int(input("Please enter the y coordinate (in monitor space 1920x1080): "))
            
            if 0 <= user_input_x <= 1920 and 0 <= user_input_y <= 1080:
                break
            else:
                print("Coordinates must be within the range (0, 0) to (1920, 1080). Please try again.")
        except ValueError:
            print("Invalid input. Please enter integer values for the coordinates.")

    dotmap_position = (user_input_x, user_input_y)
    print(f"Dotmap position: {dotmap_position}")

    return dotmap_position