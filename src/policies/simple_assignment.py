from collections import defaultdict
from src.templates import RestaurantAction, VehicleAction, Action, Observation, Policy
from src.vehicle import Stop
import copy


class SimpleAssignmentPolicy(Policy):
    r"""
    Implementation of a simple assignment policy that assigns each
    unassigned order to the vehicle with the lowest busy time.
    New stops are appended to the end of the assigned vehicle's route.
    Orders are appended to the end of the restaurant's queue.
    No postponement of order assignments is considered.
    """

    def __init__(self, tt_matrix):
        Policy.__init__(self)
        self.tt_matrix = tt_matrix



    def act(self, obs: Observation) -> Action:
        r"""
        Creates an action based on an observation.
        """
        when_to_assign = 40 * 60
        wait_time = 20 * 60

        action = {"vehicle_action": defaultdict(lambda: []),
                  "restaurant_action": defaultdict(lambda: [])}


        for customer_id, restaurant_id in obs["unassigned_orders"]:
            order_info = obs["customer_info"][customer_id]
            order_type = order_info["order_type"]
            expected_delivery_time = order_info["expected_delivery_time"]

            # Find the least busy vehicle
            vehicle_index = sorted(obs["vehicle_info"].keys(),
                                   key=lambda x: obs["vehicle_info"][x]["busy_time"])[0]

            vehicle = obs["vehicle_info"][vehicle_index]

            # # #find best vehicle
            # vehicle_index, best_insertion_index = self.get_best_vehicle_for_order(customer_id, restaurant_id, obs)
            #

            # call insertion to get insertion index
            new_insertion_index, best_cost = self.get_insertion_index_for_new_pickup_action(vehicle_index, customer_id,
                                                                                            restaurant_id, obs)
            # print(new_insertion_index)
            # print(vehicle["sequence_of_actions"])
            start_at_pre = -1
            assign_order = True
            if order_type == 'Preorder':
                order_time = int((obs["customer_info"][customer_id]['order_time']))
                etd = int(obs["customer_info"][customer_id]['expected_delivery_time'])
                start_at_pre = order_time + ((abs(order_time-etd))-wait_time)
                assign_order = (etd - obs["current_time"]) < when_to_assign




            pickup_action = VehicleAction(restaurant_id, start_at_pre, new_insertion_index + 1, [customer_id], None)
            delivery_action = VehicleAction(customer_id, -1, new_insertion_index + 2, None, [restaurant_id])

            # #For simple assignment policy run the following lines:
            # pickup_action = VehicleAction(restaurant_id, -1, -1, [customer_id], None)
            # delivery_action = VehicleAction(customer_id, -1, -1, None, [restaurant_id])

            restaurant_action = RestaurantAction(customer_id, -1, -1)
            if assign_order:
                action["vehicle_action"][vehicle_index].extend([pickup_action, delivery_action])
                action["restaurant_action"][restaurant_id].append(restaurant_action)


        return Action(action)





    def calculate_impact(self, sequence_of_actions, observation):

        delay_dict = {}
        alpha = 1
        beta = 1

        def get_customer_order_time(customer_id):
            return observation["customer_info"][customer_id]['order_time']

        def get_customer_expected_delivery_time(customer_id):
            return observation["customer_info"][customer_id]['expected_delivery_time']

        def get_tt_from_last_destination_to_restaurant(self, action, index):
            restaurant_id_key = sequence_of_actions[index + 1]["restaurant_id"]

            if restaurant_id_key is not None and restaurant_id_key in observation["restaurant_info"]:
                restaurant_id = observation["restaurant_info"][restaurant_id_key]
                result = int(self.tt_matrix[str(action["destination"])][str(restaurant_id["location"])])
                return result
            else:
                return 100  # arbitrary

        def get_order_type(customer_id):
            return observation["customer_info"][customer_id]['order_type']



        if sequence_of_actions is None or len(sequence_of_actions) == 0:
            return 0, {}
        impact = 0
        for i, action in enumerate(sequence_of_actions):
            # print(action)
            if action["type"] == "pickup":
                continue

            #delay = max(0, (action['start_at'] + action["estimated_time_required"] - get_customer_expected_delivery_time(
            #    action['customer_id'])))
            delay = max(0,
                        abs(action['start_at'] + action["estimated_time_required"] - get_customer_expected_delivery_time(
                            action['customer_id']))-10*60)



            delay_dict[action["customer_id"]] = delay
            # print(delay_dict)
            order_type = get_order_type(action['customer_id'])
            if order_type == 'Preorder':
                impact += delay * alpha
            elif order_type == 'Instant':
                impact += delay * beta
        return impact, delay_dict

    def get_estimated_waiting_time(self, orders: list, time: int, queue: list, prepared_orders: list,
                                   estimated_finish_times: list) -> float:
        r"""
        Returns the exact (not estimated) waiting time (in seconds) until a given list of orders is finished.
        """
        orders = [order for order in queue if (order["customer_id"] in orders and order not in prepared_orders)]
        if len(orders) == 0:
            return 0
        max_index = max([queue.index(order) for order in orders])
        return max(0, estimated_finish_times[max_index] - time)

    def repair_vehicle_route(self, sequence_of_stops, obs) -> None:
        r"""
        Updates travel times, waiting times, and start at times of stops after new stops have been inserted
        into the route.
        """
        estimated_time = obs["current_time"]  # estimated arrival at stop
        if sequence_of_stops[0]["started_at"] is not None:
            estimated_time = sequence_of_stops[0]["started_at"]
        for stop_index, stop in enumerate(sequence_of_stops):
            stop["start_at"] = max(stop["start_at"], estimated_time)
            estimated_time = stop["start_at"] + stop["estimated_travel_time"] + stop["estimated_park_time"]

            # for each pickup stop we adjust the wait time
            if stop["type"] == "pickup":
                restaurant_obs = obs["restaurant_info"][stop["restaurant_id"]]

                estimated_wait_time = self.get_estimated_waiting_time(stop["orders_to_pickup"], estimated_time,
                                                                      restaurant_obs["orders_in_queue"],
                                                                      restaurant_obs["prepared_orders"],
                                                                      restaurant_obs["estimated_finish_times"])
                stop["estimated_wait_time"] = estimated_wait_time
                estimated_time += estimated_wait_time
            stop["estimated_time_required"] = stop["estimated_wait_time"] + stop["estimated_travel_time"] + stop[
                "estimated_park_time"]
        return sequence_of_stops


    def get_insertion_index_for_new_pickup_action(self, vehicle_id, customer_id, restaurant_id, obs) -> (int, int):
        '''

        Parameters
        ----------
        vehicle_id
        customer_id
        restaurant_id
        obs

        Returns  (int, int) -  (Best insertion index, best cost)
        -------

        '''
        current_time = obs["current_time"]
        sequence_of_actions = obs["vehicle_info"][vehicle_id]["sequence_of_actions"]

        def get_customer_expected_delivery_time(customer_id):
            return obs["customer_info"][customer_id]['expected_delivery_time']

        # Checking if the sequence of actions is empty (added new)
        if not sequence_of_actions:
            expected_delivery_time = obs["customer_info"][customer_id]["expected_delivery_time"]
            return -1, 0
            # max(0, current_time + 2*park_times + tt_from_last_destination_to_restaurant + exo_wait_time_at_restaurant + tt_rest_customer - expected_delivery_time)

        for action in sequence_of_actions:
            if action['start_at'] == -1:


                if action["started_at"] is not None:
                    current_time = action['started_at']
                    action["start_at"] = action["started_at"]
                else:
                    action['start_at'] = current_time
                    current_time += action['estimated_time_required']
            else:
                current_time = max(current_time, action["start_at"])
            # print(action["start_at"])

        best_insertion = -1
        best_cost = float('inf')
        # delay_r = self.delay(sequence_of_actions, obs)
        cost_r, old_delay_dict = self.calculate_impact(sequence_of_actions, obs)
        for i, action in enumerate(sequence_of_actions):

            if action["type"] == "pickup" or action["started_at"] is not None:
                continue
            # Create a copy of the current sequence and try inserting the pickup action
            new_route = copy.deepcopy(sequence_of_actions)
            estimated_travel_time_pickup = int(
                self.tt_matrix[str(action["destination"])][str(obs["restaurant_info"][restaurant_id]["location"])])

            estimated_travel_time_delivery = int(self.tt_matrix[str(obs["restaurant_info"][restaurant_id]["location"])][
                                                     str(obs["customer_info"][customer_id]["location"])])
            new_pickup_stop = Stop("pickup", obs["restaurant_info"][restaurant_id]["location"], restaurant_id,
                                   None, -1,
                                   estimated_travel_time_pickup, estimated_travel_time_pickup,
                                   162, 162,
                                   -1, -1,
                                   [customer_id]).summary()
            new_delivery_stop = Stop("delivery", obs["customer_info"][customer_id]["location"], None,
                                     customer_id, -1,
                                     estimated_travel_time_delivery, estimated_travel_time_delivery,
                                     162, 162,
                                     0, 0,
                                     None).summary()

            new_route.insert(i + 1, new_pickup_stop)
            new_route.insert(i + 2, new_delivery_stop)

            # repair
            new_route = self.repair_vehicle_route(new_route, obs)
            # print(new_route)

            # # Calculate the insertion cost
            # insertion_cost = self.delay(new_route, obs) - cost_r
            new_impact, new_delay_dict = self.calculate_impact(new_route, obs)
            insertion_cost = new_impact - cost_r
            # print(sequence_of_actions, new_route)

            # print(new_impact, cost_r)

            insertion_feasible = True
            # print(old_delay_dict, new_delay_dict)
            for c_id in old_delay_dict.keys():

                if (1800 < old_delay_dict[c_id] < new_delay_dict[c_id]) or (new_delay_dict[c_id]<old_delay_dict[c_id]<-1800):
                    # print(old_delay_dict[c_id] , new_delay_dict[c_id], "NOT FEASIBLE")
                    insertion_feasible = False
                    break

            # Update the best cost and best insertion index if this is the best option so far
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_insertion = i
        return best_insertion, best_cost
