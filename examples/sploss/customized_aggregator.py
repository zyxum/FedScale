from fedscale.core.aggregator import Aggregator
from customized_init_model import customized_init_model
from fedscale.core.fl_aggregator_libs import logDir
from fedscale.core.arg_parser import args
from fedscale.core import events
import pickle, logging, os

class Customized_Aggregator(Aggregator):
    def init_model(self):
        self.model =  customized_init_model()
        self.model_weights = self.model.state_dict()
    
    def testing_completion_handler(self, client_id, results):
        """Each executor will handle a subset of testing dataset
        """
        results = results['results']
        if results == None:
            logging.info("skip testing completion")
            self.broadcast_events_queue.append(events.START_ROUND)
            return

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
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock,
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
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'sp_loss': accumulator['sp_loss'],
                    'test_len': accumulator['test_len']
                    }


            logging.info("FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, sploss: {}, test loss: {:.4f}, test len: {}"
                    .format(self.round, self.global_virtual_clock, self.testing_history['perf'][self.round]['top_1'],
                    self.testing_history['perf'][self.round]['top_5'], self.testing_history['perf'][self.round]['sp_loss'],self.testing_history['perf'][self.round]['loss'],
                    self.testing_history['perf'][self.round]['test_len']))

            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf'), 'wb') as fout:
                pickle.dump(self.testing_history, fout)

            if len(self.loss_accumulator):
                self.log_writer.add_scalar('Test/round_to_loss', self.testing_history['perf'][self.round]['loss'], self.round)
                self.log_writer.add_scalar('Test/round_to_accuracy', self.testing_history['perf'][self.round]['top_1'], self.round)
                self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.round]['loss'],
                                            self.global_virtual_clock/60.)
                self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.round]['top_1'],
                                            self.global_virtual_clock/60.)
                self.log_writer.add_scalars('Test/sp_loss', self.testing_history['perf'][self.round]['sp_loss'], self.round)

            self.broadcast_events_queue.append(events.START_ROUND)

if __name__ == "__main__":
    aggregator = Customized_Aggregator(args)
    aggregator.run()