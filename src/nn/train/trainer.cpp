
#include <fstream>
#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    Trainer::Trainer(
        Module* _model,
        optimizer::Optimizer* _optimizer,
        std::function<float(std::vector<Tensor>,Tensor)> _loss_fn,
        int64_t _batch_size,
        data::DataLoader*_train_loader,
        std::vector<Callback*> _callbacks,
        data::DataLoader* _val_loader,
        trainer_state _states 
    ):
        model(_model),
        optimizer(_optimizer),
        loss_fn(_loss_fn),
        batch_size(_batch_size),
        train_loader(train_loader),
        val_loader(_val_loader),
        callbacks(_callbacks),
        states(_states)
    {}

    void  Trainer::fit(int64_t num_epochs){

        _trigger_callbacks("on_train_begin");

        for (int64_t epoch = 1; epoch <= num_epochs; epoch++){

            _trigger_callbacks("on_epoch_begin");

            states.epoch = epoch;
            states.

            _train_epoch();

            if (val_loader) _validate_epoch();

            _trigger_callbacks("on_epoch_end");

            if (states.stop_training) break;

        }

        _trigger_callbacks("on_train_end");

    }

    void Trainer::_train_epoch(){

        int64_t i = 0;
        float loss = 0;

        while ( train_loader->has_next() ){
            
            states.step_in_epoch = ++i;
            states.global_step++;

            _trigger_callbacks("on_batch_begin");

            auto batch = train_loader->next();

            loss +=  states.last_train_loss = _train_step(batch);

            _trigger_callbacks("on_batch_end");


        }

        train_loader->reset();

        states.train_loss = loss / batch_size;

    }


    void Trainer::_validate_epoch(){

        int64_t i = 0;
        float loss = 0;

        while ( val_loader->has_next() ){

            auto batch = val_loader->next();

            loss += _train_step(batch);

        }

        states.val_loss = loss / batch_size;

        val_loader->reset();

    }

    float  Trainer::_train_step(const std::pair<Tensor,Tensor>& batch){

        auto [input, target] = batch;
        
        auto outputs = model->forward(input);
        auto loss = loss_fn(outputs, );
        

    }

    float  Trainer::_eval_step(const std::pair<Tensor,Tensor>& batch);

}//nn
}//mt
