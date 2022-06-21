# za iscrtavanje gradika tacnosti i f-je gubitaka (promene u kodu)

accuracy =[]
loss =[]

class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, api):
        self.my_monitor_api = api

#def on_batch_end(training_state, snapshot, log={}):    
    def on_sub_batch_end(self, training_state, train_index=0):
        try:
            accuracy.append( str(training_state.acc_value) )
            loss.append( str(training_state.loss_value) )
        except Exception as e:
            print(str(e))

monitorCallback = MonitorCallback(tflearn)

#dodata je callback f-ja koja u niz accuracy i niz loss dodaje posle svake iteracije vrednosti
#posle treniranja se u pozivanju fje fit dodaje callback

model.fit(X_train, Y_train, n_epoch=150, batch_size=1, show_metric=True,callbacks=monitorCallback)

import matplotlib.pyplot as plt
plt.plot( accuracy )
plt.plot( loss )
plt.show()
plt.close()
#isrctavanje grafika
