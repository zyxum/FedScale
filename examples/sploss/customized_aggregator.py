from fedscale.core.aggregator import Aggregator
from customized_init_model import customized_init_model
from fedscale.core.fl_aggregator_libs import logDir
from fedscale.core.arg_parser import args
import pickle, logging, os

class Customized_Aggregator(Aggregator):
    def init_model(self):
        return customized_init_model()
    
    def testing_completion_handler(self, responses):
        """Each executor will handle a subset of testing dataset
        """
        response = pickle.loads(responses.result().serialized_test_response)
        executorId, results = response['executorId'], response['results']

        # List append is thread-safe
        self.test_result_accumulator.append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator) == len(self.executors):
            accumulator = self.test_result_accumulator[0]
            for i in range(1, len(self.test_result_accumulator)):
                if self.args.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(self.imdb.num_classes):
                                accumulator[key][j] = accumulator[key][j] + self.test_result_accumulator[i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[i][key]
                else:
                    for key in accumulator:
                        accumulator[key] += self.test_result_accumulator[i][key]
            if self.args.task == "detection":
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator), 4),
                    'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator), 4),
                    'loss': accumulator['test_loss'],
                    'test_len': accumulator['test_len']
                    }
            else:
                sp_loss_list = accumulator['sp_loss']
                accumulator['sp_loss'] = {}
                for i, layer_loss in enumerate(sp_loss_list):
                    accumulator['sp_loss']['layer'+str(i)] = layer_loss
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'sp_loss': accumulator['sp_loss'],
                    'test_len': accumulator['test_len']
                    }


            logging.info("FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, sploss: {}, test loss: {:.4f}, test len: {}"
                    .format(self.epoch, self.global_virtual_clock, self.testing_history['perf'][self.epoch]['top_1'],
                    self.testing_history['perf'][self.epoch]['top_5'], self.testing_history['perf'][self.epoch]['sp_loss'],self.testing_history['perf'][self.epoch]['loss'],
                    self.testing_history['perf'][self.epoch]['test_len']))

            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf'), 'wb') as fout:
                pickle.dump(self.testing_history, fout)

            if len(self.loss_accumulator):
                self.log_writer.add_scalar('Test/round_to_loss', self.testing_history['perf'][self.epoch]['loss'], self.epoch)
                self.log_writer.add_scalar('Test/round_to_accuracy', self.testing_history['perf'][self.epoch]['top_1'], self.epoch)
                self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.epoch]['loss'],
                                            self.global_virtual_clock/60.)
                self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.epoch]['top_1'],
                                            self.global_virtual_clock/60.)
                self.log_writer.add_scalars('Test/sp_loss', self.testing_history['perf'][self.epoch]['sp_loss'], self.epoch)

            self.event_queue.append('start_round')

if __name__ == "__main__":
    aggregator = Customized_Aggregator(args)
    aggregator.run()