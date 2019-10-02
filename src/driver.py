
import src.datasets.data_set as ds


def get_abalone_data():
    abalone_data = ds.DataSet("../data/abalone.data", 9, list(range(0, 9)))
    # Convert attribute columns to floats
    abalone_data.convert_to_float(abalone_data.attr_cols)
    # Normalize values
    abalone_data.normalize_z_score(list(range(1, 9)))
    return abalone_data


def get_car_data():
    car_data = ds.DataSet("../data/car.data", 6, list(range(0, 6)))
    # Convert attribute columns to numeric scheme
    car_data.convert_attribute(0, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    car_data.convert_attribute(1, {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    car_data.convert_attribute(2, {'2': 2, '3': 3, '4': 4, '5more': 5})
    car_data.convert_attribute(3, {'2': 2, '4': 4, 'more': 5})
    car_data.convert_attribute(4, {'small': 0, 'med': 1, 'big': 2})
    car_data.convert_attribute(5, {'low': 0, 'med': 1, 'high': 2})
    # Normalize values.
    car_data.normalize_z_score(list(range(0, 6)))
    return car_data


def get_forest_fires_data():
    forest_fires_data = ds.DataSet("../data/forestfires.data", 12, list(range(0, 12)))
    # Remove the first line, which is the header info.
    forest_fires_data.remove_header()
    # Convert all columns to floats, including the class column.
    forest_fires_data.convert_to_float(list(range(0, 13)))
    return forest_fires_data


def get_machine_data():
    # There is another final column but we probably want to exclude it.
    machine_data = ds.DataSet("../data/machine.data", 8, list(range(0, 8)))
    # Convert all columns except the first two to floats, including the class column.
    machine_data.convert_to_float(list(range(2, 9)))
    return machine_data


def main():
    abalone_data = get_abalone_data()
    car_data = get_car_data()
    forest_fires_data = get_forest_fires_data()
    machine_data = get_machine_data()

main()