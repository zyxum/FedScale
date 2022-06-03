from fedscale.core.utils.divide_data import DataPartitioner
import logging, csv
class Customized_DataPartitioner(DataPartitioner):
    def trace_partition_with_IDmap(self, data_map_file, unique_clientIds):
        logging.info(f"Partitioning data by profile {data_map_file}...")

        self.partitions = [[] for _ in range(len(unique_clientIds))]
        clientId_maps = {}

        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    clientId_maps[sample_id] = unique_clientIds[client_id]
                    self.client_label_cnt[unique_clientIds[client_id]].add(row[-1])
                    sample_id += 1
        
        for idx in range(sample_id):
            self.partitions[clientId_maps[idx]].append(idx)
    
    def trace_partition(self, data_map_file):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")

        clientId_maps = {}
        unique_clientIds = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_clientIds:
                        unique_clientIds[client_id] = len(unique_clientIds)

                    clientId_maps[sample_id] = unique_clientIds[client_id]
                    self.client_label_cnt[unique_clientIds[client_id]].add(row[-1])
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]

        for idx in range(sample_id):
            self.partitions[clientId_maps[idx]].append(idx)
        return unique_clientIds

    def partition_data_helper(self, num_clients, data_map_file=None, unique_clientId=None):
        # read mapping file to partition trace
        if unique_clientId is None:
            if data_map_file is not None:
                unique_clientId = self.trace_partition(data_map_file)
                return unique_clientId
            else:
                self.uniform_partition(num_clients=num_clients)
        else:
            self.trace_partition_with_IDmap(data_map_file, unique_clientId)