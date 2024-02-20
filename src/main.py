from src.state import MealDeliveryMDP
from src.policies.simple_assignment import SimpleAssignmentPolicy
import configparser
import json
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

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

    for i in range(0, 1):
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
                # summarizing preorders-
                pre_order_delays = []
                customer_data = []
                number_of_delayed_orders = 0

                instant_order_delays = []
                instant_customer_data = []
                number_of_delayed_instant_orders = 0

                # Loop through each customer in served requests
                for customer in env.served_requests:
                    if customer.order_type == "Preorder":
                        # Calculate delay
                        p_delay = int(max(customer.delivery_time.values()) - customer.expected_delivery_time)
                        pre_order_delays.append(p_delay)

                        if p_delay > 0:
                            number_of_delayed_orders += 1

                        # order time, expected delivery time, and actual delivery time
                        order_time = customer.order_time
                        expected_delivery_time = int(customer.expected_delivery_time)
                        actual_delivery_time = max(
                            customer.delivery_time.values())

                        customer_data.append({
                            "Order Time": order_time,
                            "Expected Delivery Time": expected_delivery_time,
                            "Actual Delivery Time": actual_delivery_time,
                            "Delay": p_delay
                        })
                df = pd.DataFrame(customer_data)
                print(df)
                # barchart
                df.plot(kind='bar', x='Order Time', y='Delay', color='blue')
                plt.xlabel('Order Time')
                plt.ylabel('Delay (seconds)')
                plt.title('Preorder Delays Visualization')
                plt.show()
                #preorder mean delay
                if len(pre_order_delays) > 0:
                    mean_preorder_delay = sum(pre_order_delays) / len(pre_order_delays) / 60
                    mean_preorder_delay = round(mean_preorder_delay, 2)
                    print(f"Mean Preorder Delay: {mean_preorder_delay} minutes")
                else:
                    print("No Preorder Delays to calculate mean.")

                print(f"Number of Delayed Orders: {number_of_delayed_orders}")

                # ##########################################################################################
                # ####### instant_order_summary
                # instant_order_delays = []
                # instant_customer_data = []
                # number_of_delayed_instant_orders = 0
                #
                # # Loop through each customer in served requests
                # for customer in env.served_requests:
                #     if customer.order_type == "Instant":
                #         # Calculate delay
                #         i_delay = int(max(0, max(customer.delivery_time.values()) - customer.expected_delivery_time))
                #         instant_order_delays.append(i_delay)
                #
                #         if i_delay > 0:
                #             number_of_delayed_instant_orders += 1
                #
                #         # order time, expected delivery time, and actual delivery time
                #         order_time = customer.order_time
                #         expected_delivery_time = int(customer.expected_delivery_time)
                #         actual_delivery_time = max(customer.delivery_time.values())
                #
                #         instant_customer_data.append({
                #             "Order Time": order_time,
                #             "Expected Delivery Time": expected_delivery_time,
                #             "Actual Delivery Time": actual_delivery_time,
                #             "Delay": i_delay
                #         })
                #
                # instant_df = pd.DataFrame(instant_customer_data)
                # print(instant_df)
                #
                # # Barchart for instant orders
                # instant_df.plot(kind='bar', x='Order Time', y='Delay', color='red')
                # plt.xlabel('Order Time')
                # plt.ylabel('Delay (seconds)')
                # plt.title('Instant Order Delays Visualization')
                # plt.show()
                #
                # # Instant order mean delay
                # if len(instant_order_delays) > 0:
                #     mean_instant_delay = sum(instant_order_delays) / len(instant_order_delays) / 60
                #     mean_instant_delay = round(mean_instant_delay, 2)
                #     print(f"Mean Instant Order Delay: {mean_instant_delay} minutes")
                # else:
                #     print("No Instant Order Delays to calculate mean.")
                #
                # print(f"Number of Delayed Instant Orders: {number_of_delayed_instant_orders}")
                #
                # # Export the DataFrame to an Excel file
                # # instant_df.to_excel('instant_order_data.xlsx', index=False)
                #
                # instant_delays = [max(0, max(customer.delivery_time.values()) - customer.expected_delivery_time)
                #                     for customer in env.served_requests if customer.order_type == "pre_order"]
                # print("Episode {}. Mean delay {}.".format(i, env.mean_delay))

                print("Episode {}: Mean delay is {}.".format(i + 1, env.mean_delay))
                break
