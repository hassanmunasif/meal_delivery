from src.state import MealDeliveryMDP
from src.policies.simple_assignment import SimpleAssignmentPolicy
import configparser
import json
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
# pd.set_option('display.max_rows', None)
print(MealDeliveryMDP)


def default_serializer(obj):
    """Convert non-serializable objects for JSON."""
    if isinstance(obj, np.int32):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


if __name__ == "__main__":

    config = configparser.ConfigParser(allow_no_value=True)
    config.read('H:/meal_delivery/abc/data/instances/iowa_110_5_55_80.ini')
    env = MealDeliveryMDP(config, seed=42)
    policy = SimpleAssignmentPolicy(tt_matrix=env.tt_matrix)

    delays = []
    preorder_delays_accumulated = []
    instant_delays_accumulated = []
    preorder_accuracy_accumulated = []
    instant_accuracy_accumulated = []
    overall_accuracy_accumulated = []

    for i in range(0, 5):
        try:
            obs = env.reset()
            while True:
                action = policy.act(obs)
                obs, cost, done, info = env.step(action)

                # vehicle rout_formatted
                # try:
                #     sequence = json.dumps(obs["vehicle_info"]["v_4"]["sequence_of_actions"],
                #                           indent=4, default=default_serializer)
                #     print("Sequence of Actions for Vehicle v_0:\n", sequence)
                # except TypeError as e:
                #     print("Error serializing sequence_of_actions:", e)
                #
                # print(obs["vehicle_info"]["v_3"]["sequence_of_actions"])

                if done:
                    # summarizing preorders
                    pre_order_delays = []
                    customer_data = []
                    number_of_delayed_orders = 0

                    # Loop through each customer in served requests
                    for customer in env.served_requests:
                        if customer.order_type == "Preorder":
                            actual_delivery_time = max(customer.delivery_time.values())
                            expected_delivery_time = int(customer.expected_delivery_time)
                            actual_delay = actual_delivery_time - expected_delivery_time

                            # Implementing new delay logic
                            if actual_delay > 10 * 60:  # More than 5 minutes late
                                delay = actual_delay - 10*60
                            elif actual_delay < -10 * 60:  # More than 10 minutes early
                                delay = -actual_delay - 10*60
                            else:
                                delay = 0  # Within the acceptable window

                            # Convert delay from seconds to minutes
                            delay_in_minutes = delay

                            pre_order_delays.append(delay)
                            if delay != 0:
                                number_of_delayed_orders += 1

                            customer_data.append({
                                "Order Time": customer.order_time,
                                "Order Types": customer.order_type,
                                "Expected Delivery Time": expected_delivery_time,
                                "Actual Delivery Time": actual_delivery_time,
                                "Delay (minutes)": delay_in_minutes
                            })
                    # Generate DataFrame from the collected customer data
                    df = pd.DataFrame(customer_data)
                    # print(df)
                    # barchart
                    # df.plot(kind='bar', x='Order Time', y='Delay', color='blue')
                    # plt.xlabel('Order Time')
                    # plt.ylabel('Delay (seconds)')
                    # plt.title('Preorder Delays Visualization')
                    # plt.show()
                    #preorder mean delay
                    if pre_order_delays:
                        mean_preorder_delay = sum(pre_order_delays) / len(pre_order_delays) / 60
                        mean_preorder_delay = round(mean_preorder_delay, 2)
                        # print(f"Mean Preorder Delay: {mean_preorder_delay} minutes")
                        preorder_delays_accumulated.append(mean_preorder_delay)

                        # Calculate and print the accuracy
                    total_pre_orders = len(pre_order_delays)
                    if total_pre_orders > 0:
                        accuracy = 1 - (number_of_delayed_orders / total_pre_orders)
                        # print(f"Accuracy (Percentage of On-Time pre-Orders): {accuracy:.2%}")
                        preorder_accuracy_accumulated.append(accuracy)

                    # print(f"Number of Delayed Orders: {number_of_delayed_orders}")

                    # ##########################################################################################
                    ##### instant_order_summary
                    instant_order_delays = []
                    instant_customer_data = []
                    number_of_delayed_instant_orders = 0

                    # Loop through each customer in served requests
                    for customer in env.served_requests:
                        if customer.order_type == "Instant":
                            actual_delivery_time = max(customer.delivery_time.values())
                            expected_delivery_time = int(customer.expected_delivery_time)
                            actual_delay = actual_delivery_time - expected_delivery_time

                            # Implementing new delay logic for instant orders
                            if actual_delay > 10 * 60:  # More than 5 minutes late
                                delay = actual_delay - 10*60
                            elif actual_delay < -10 * 60:  # More than 10 minutes early
                                delay = -actual_delay - 10*60
                            else:
                                delay = 0  # Within the acceptable window

                            instant_order_delays.append(delay)
                            if delay != 0:
                                number_of_delayed_instant_orders += 1

                            delay_in_minutes = delay

                            instant_customer_data.append({
                                "Order Time": customer.order_time,

                                "Expected Delivery Time": expected_delivery_time,
                                "Actual Delivery Time": actual_delivery_time,
                                "Delay (minutes)": delay_in_minutes

                            })

                    # Generate DataFrame from the collected instant customer data
                    instant_df = pd.DataFrame(instant_customer_data)
                    # print(instant_df)

                    # Barchart for instant orders
                    # instant_df.plot(kind='bar', x='Order Time', y='Delay', color='red')
                    # plt.xlabel('Order Time')
                    # plt.ylabel('Delay (seconds)')
                    # plt.title('Instant Order Delays Visualization')
                    # plt.show()

                    if instant_order_delays:
                        mean_instant_delay = sum(instant_order_delays) / len(instant_order_delays) / 60
                        mean_instant_delay = round(mean_instant_delay, 2)
                        # print(f"Mean Instant Order Delay: {mean_instant_delay} minutes")
                        instant_delays_accumulated.append(mean_instant_delay)

                        # Calculate and print the accuracy for instant orders
                    total_instant_orders = len(instant_order_delays)
                    if total_instant_orders > 0:
                        accuracy_instant_orders = 1 - (number_of_delayed_instant_orders / total_instant_orders)
                        # print(f"Accuracy (Percentage of On-Time Instant Orders): {accuracy_instant_orders:.2%}")
                        instant_accuracy_accumulated.append(accuracy_instant_orders)

                    else:
                        print("No instant orders to calculate accuracy.")

                    # print(f"Number of Delayed Instant Orders: {number_of_delayed_instant_orders}")

                    # Export the DataFrame to an Excel file
                    # instant_df.to_excel('instant_order_data.xlsx', index=False)

                    # Calculate and print the overall accuracy
                    total_orders = total_pre_orders + total_instant_orders  # Total number of both preorder and instant orders
                    total_delayed_orders = number_of_delayed_orders + number_of_delayed_instant_orders  # Total number of delayed preorder and instant orders
                    if total_orders > 0:
                        overall_accuracy = 1 - (total_delayed_orders / total_orders)
                        # print(f"Overall Accuracy (Percentage of On-Time Orders): {overall_accuracy:.2%}")
                        overall_accuracy_accumulated.append(overall_accuracy)

                    else:
                        print("No orders to calculate overall accuracy.")



                    # instant_delays = [max(0, max(customer.delivery_time.values()) - customer.expected_delivery_time)
                    #                     for customer in env.served_requests if customer.order_type == "pre_order"]


                    print("Episode {}: Mean delay is {}.".format(i + 1, env.mean_delay))
                    delays.append(env.mean_delay)
                    break
        except Exception as e:
            print(f"Error occurred during Episode {i}: {str(e)}")
    if delays:  # Ensure there are delays recorded to avoid division by zero
        average_delay = sum(delays) / len(delays)
        print("Average mean delay over all episodes: {:.3f}".format(average_delay))
    else:
        print("No episodes were completed, so average delay cannot be calculated.")
    if preorder_delays_accumulated:  # Calculate average preorder delay
        average_preorder_delay = sum(preorder_delays_accumulated) / len(preorder_delays_accumulated)
        print(f"Average Preorder Mean Delay over all episodes: {average_preorder_delay:.3f} minutes")
    else:
        print("No preorder episodes were completed, so average delay cannot be calculated.")
    if instant_delays_accumulated:  # Calculate average instant delay
        average_instant_delay = sum(instant_delays_accumulated) / len(instant_delays_accumulated)
        print(f"Average Instant Mean Delay over all episodes: {average_instant_delay:.3f} minutes")
    else:
        print("No instant episodes were completed, so average delay cannot be calculated.")
    if preorder_accuracy_accumulated:  # Calculate average preorder accuracy
        average_preorder_accuracy = sum(preorder_accuracy_accumulated) / len(preorder_accuracy_accumulated)
        print(f"Average Preorder Accuracy over all episodes: {average_preorder_accuracy:.2%}")
    else:
        print("No preorder accuracy data to calculate.")

    if instant_accuracy_accumulated:  # Calculate average instant order accuracy
        average_instant_accuracy = sum(instant_accuracy_accumulated) / len(instant_accuracy_accumulated)
        print(f"Average Instant Order Accuracy over all episodes: {average_instant_accuracy:.2%}")
    else:
        print("No instant order accuracy data to calculate.")

    if overall_accuracy_accumulated:  # Calculate average overall accuracy
        average_overall_accuracy = sum(overall_accuracy_accumulated) / len(overall_accuracy_accumulated)
        print(f"Average Overall Accuracy over all episodes: {average_overall_accuracy:.2%}")
    else:
        print("No overall accuracy data to calculate.")