#ifndef NN_HPP 
#define NN_HPP
#include <vector>
#include <stdexcept>
#include <functional>
#include <initializer_list>
#include <config/mtensor_export.hpp>
#include <MTensor/tensor.hpp>
#include <MTensor/data.hpp>

namespace mt {
namespace nn {


    //core
    class MTENSOR_API Module {
    public:
        virtual Tensor forward(Tensor input) = 0;
        virtual Tensor operator()(Tensor input);
        //virtual Tensor backward(const Tensor& grad_output) = 0;

        std::vector<Tensor> paramters() const;
        std::vector<Tensor> auxiliaries() const;
        Tensor auxiliary(const Tensor& aux);
        void auxiliary(const std::vector<Tensor>& aux);

        virtual ~Module() = default;

    protected:

        std::shared_ptr<Module> module(const std::shared_ptr<Module>& module);
        Tensor paramter(const Tensor& prm);

        
        std::vector<std::shared_ptr<Module>> m_direct_children; 
        std::vector<Tensor> m_direct_paramters;
        std::vector<Tensor> m_direct_auxiliaries;


    private:

        void _parameters( std::vector<Tensor>& collector ) const;
        void _auxiliaries( std::vector<Tensor>& collector ) const;

    };

    using m = std::shared_ptr<Module>;


    class MTENSOR_API SequentialImpl: public Module {

    public:
        SequentialImpl(std::initializer_list<std::shared_ptr<Module>> list);

        Tensor forward(Tensor x) override;
    };

    MTENSOR_API std::shared_ptr<Module> Sequential(std::initializer_list<std::shared_ptr<Module>> list);


    ////////////////////////////////////////////////////////////////////
    
    //layers

    class MTENSOR_API LinearImpl: public Module{

        Tensor m_weights;
        Tensor m_bias;
        ops::Matmul matmul;

    public: 

        LinearImpl(int64_t in_features, int64_t out_features, bool bias = true, int64_t seed = -1);
        Tensor forward(Tensor input) override;
        Tensor weights() const;
        Tensor bias() const;

    };

    MTENSOR_API std::shared_ptr<Module> Linear(int64_t in_features, int64_t out_features, bool bias = true, int64_t seed = -1);

    
    class MTENSOR_API Conv1dImpl: public Module{

        ops::Conv1d conv1d;
        Tensor weights;
        Tensor bias;

    public: 

        Conv1dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size, bool bias = true, int64_t stride = 1, int64_t padding_l = 0, int64_t padding_r = 0);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Conv1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, bool bias = true, int64_t stride = 1, int64_t padding_l = 0, int64_t padding_r = 0);

    class MTENSOR_API Conv2dImpl: public Module{

        ops::Conv2d conv2d;
        Tensor weights;
        Tensor bias;

    public: 

        Conv2dImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool bias = true, const std::vector<int64_t>& stride = {1,1}, const std::vector<int64_t>& padding_l = {0,0}, const std::vector<int64_t>& padding_r = {0,0});
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Conv2d(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool bias = true, const std::vector<int64_t>& stride = {1,1}, const std::vector<int64_t>& padding_l = {0,0}, const std::vector<int64_t>& padding_r = {0,0});


    class MTENSOR_API Conv3dImpl: public Module{

        ops::Conv3d conv3d;
        Tensor weights;
        Tensor bias;

    public: 

        Conv3dImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool bias = true, const std::vector<int64_t>& stride = {1,1,1}, const std::vector<int64_t>& padding_l = {0,0,0}, const std::vector<int64_t>& padding_r = {0,0,0});
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Conv3d(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool bias = true, const std::vector<int64_t>& stride = {1,1,1}, const std::vector<int64_t>& padding_l = {0,0,0}, const std::vector<int64_t>& padding_r = {0,0,0});



    class MTENSOR_API ConvTranspose1dImpl: public Module{

        ops::Deconv1d convTranspose1d;
        Tensor weights;
        Tensor bias;

    public: 

        ConvTranspose1dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size, bool bias = true, int64_t stride = 1, int64_t padding_l = 0, int64_t padding_r = 0);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> ConvTranspose1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, bool bias = true, int64_t stride = 1, int64_t padding_l = 0, int64_t padding_r = 0);


    class MTENSOR_API ConvTranspose2dImpl: public Module{

        ops::Deconv2d convTranspose2d;
        Tensor weights;
        Tensor bias;

    public: 

        ConvTranspose2dImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool bias = true, const std::vector<int64_t>& stride = {1,1}, const std::vector<int64_t>& padding_l = {0,0}, const std::vector<int64_t>& padding_r = {0,0});
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> ConvTranspose2d(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool bias = true, const std::vector<int64_t>& stride = {1,1}, const std::vector<int64_t>& padding_l = {0,0}, const std::vector<int64_t>& padding_r = {0,0});


    class MTENSOR_API ConvTranspose3dImpl: public Module{

        ops::Deconv3d convTranspose3d;
        Tensor weights;
        Tensor bias;

    public: 

        ConvTranspose3dImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool bias = true, const std::vector<int64_t>& stride = {1,1,1}, const std::vector<int64_t>& padding_l = {0,0,0}, const std::vector<int64_t>& padding_r = {0,0,0});
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> ConvTranspose3d(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool bias = true, const std::vector<int64_t>& stride = {1,1,1}, const std::vector<int64_t>& padding_l = {0,0,0}, const std::vector<int64_t>& padding_r = {0,0,0});



    class MTENSOR_API MaxPooling1dImpl: public Module{

        ops::MaxPooling1d maxpooling1d;

    public: 

        MaxPooling1dImpl(int64_t pooling_size, int64_t stride, int64_t padding_l = 0, int64_t padding_r = 0);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> MaxPooling1d(int64_t pooling_size, int64_t stride, int64_t padding_l = 0, int64_t padding_r = 0);



    class MTENSOR_API MaxPooling2dImpl: public Module{

        ops::MaxPooling2d maxpooling2d;

    public: 

        MaxPooling2dImpl(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding_l = {0,0}, const std::vector<int64_t>& padding_r = {0,0});
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> MaxPooling2d(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding_l = {0,0}, const std::vector<int64_t>& padding_r = {0,0});


    class MTENSOR_API MaxPooling3dImpl: public Module{

        ops::MaxPooling3d maxpooling3d;

    public: 

        MaxPooling3dImpl(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding_l = {0,0,0}, const std::vector<int64_t>& padding_r = {0,0,0});
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> MaxPooling3d(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding_l = {0,0,0}, const std::vector<int64_t>& padding_r = {0,0,0});


    class MTENSOR_API AvgPooling1dImpl: public Module{

        ops::AvgPooling1d avgpooling1d;

    public: 

        AvgPooling1dImpl(int64_t pooling_size, int64_t stride, int64_t padding_l = 0, int64_t padding_r = 0, bool include_padding = false);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> AvgPooling1d(int64_t pooling_size, int64_t stride, int64_t padding_l = 0, int64_t padding_r = 0, bool include_padding = false);


    class MTENSOR_API AvgPooling2dImpl: public Module{

        ops::AvgPooling2d avgpooling2d;

    public: 

        AvgPooling2dImpl(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding_l = {0,0}, const std::vector<int64_t>& padding_r = {0,0}, bool include_padding = false);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> AvgPooling2d(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding_l = {0,0}, const std::vector<int64_t>& padding_r = {0,0}, bool include_padding = false);


    class MTENSOR_API AvgPooling3dImpl: public Module{

        ops::AvgPooling3d avgpooling3d;

    public: 

        AvgPooling3dImpl(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding_l = {0,0,0}, const std::vector<int64_t>& padding_r = {0,0,0}, bool include_padding = false);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> AvgPooling3d(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding_l = {0,0,0}, const std::vector<int64_t>& padding_r = {0,0,0}, bool include_padding = false);


    class MTENSOR_API BatchNormImpl: public Module{

        const bool& m_training;
        float m_momentum;
        Tensor running_mean;
        Tensor running_var;
        Tensor gamma; 
        Tensor beta;

    public: 

        BatchNormImpl(int64_t num_features, bool& training, float momentum = 0.1f );
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> BatchNorm(int64_t num_features, bool& training ,float momentum = 0.1f);


    class MTENSOR_API LayerNormImpl: public Module{

        ops::LayerNormalization ln;
        Tensor gamma; 
        Tensor beta;

    public: 

        LayerNormImpl(int64_t num_features);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> LayerNorm(int64_t num_features);


    class MTENSOR_API RMSNormImpl: public Module{

        ops::RMSNormalization rmsn;
        Tensor gamma; 

    public: 

        RMSNormImpl(int64_t num_features);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> RMSNorm(int64_t num_features);

    class MTENSOR_API GroupNormImpl: public Module{

        ops::GroupNormalization gn;
        Tensor gamma; 
        Tensor beta;

    public: 

        GroupNormImpl(int64_t groups, int64_t num_channels);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> GroupNorm(int64_t num_features);

    
    class MTENSOR_API FlattenImpl: public Module{

    public: 

        FlattenImpl();
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Flatten();
    

    class MTENSOR_API UnflattenImpl: public Module{

        Dim _dim;
        Shape _shape;
        Dim numel;

    public: 

        UnflattenImpl(Dim dim, Shape shape);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Unflatten(Dim dim, Shape shape);

    class MTENSOR_API EmbeddingImpl: public Module{

        Tensor embeddings;

    public: 

        EmbeddingImpl(int64_t num_embedding, int64_t embedding_dim);
        Tensor forward(Tensor input) override;
        Tensor weights() const;

    };

    MTENSOR_API std::shared_ptr<Module> Embedding(int64_t num_embedding, int64_t embedding_dim);

    class MTENSOR_API Dropout1dImpl: public Module{

        float m_p;
        ops::Mul mul;

    public: 

        Dropout1dImpl(float p);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Dropout1d(float p);


    class MTENSOR_API Dropout2dImpl: public Module{

        float m_p;
        ops::Mul mul;

    public: 

        Dropout2dImpl(float p);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Dropout2d(float p);


    class MTENSOR_API Dropout3dImpl: public Module{

        float m_p;
        ops::Mul mul;

    public: 

        Dropout3dImpl(float p);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Dropout3d(float p);

    ////////////////////////////////////////////////////////////////////
    //activations

    class MTENSOR_API ReluImpl: public Module{

        ops::Relu relu;

    public: 

        ReluImpl(float a = 0.0f);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Relu(float a = 0.0f);
    
    class MTENSOR_API SigmoidImpl: public Module{

        ops::Sigmoid sigmoid;

    public: 

        SigmoidImpl();
        Tensor forward(Tensor input) override;

    };
    MTENSOR_API std::shared_ptr<Module> Sigmoid();

    class MTENSOR_API TanhImpl: public Module{
        
        ops::Tanh tanh;

    public: 

        TanhImpl();
        Tensor forward(Tensor input) override;

    };
    
    MTENSOR_API std::shared_ptr<Module> Tanh();


    class MTENSOR_API SoftmaxImpl: public Module{
        
        ops::Softmax softmax;

    public: 

        SoftmaxImpl(Dim dim);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> Softmax(Dim dim);


    class MTENSOR_API LogSoftmaxImpl: public Module{
        
        ops::SoftmaxLog logsoftmax;

    public: 

        LogSoftmaxImpl(Dim dim);
        Tensor forward(Tensor input) override;

    };

    MTENSOR_API std::shared_ptr<Module> LogSoftmax(Dim dim);


    ///////////////////////////////////////////////////////////////////
    //losses 


    //NOTE: for now these are implemented using the already implemented primitives, 
    //      i will switch to fused_kernels later as future improvement 

    class MTENSOR_API Loss {

    public:
        virtual Tensor forward(Tensor output, Tensor target) = 0;
        Tensor operator()(Tensor output, Tensor target){
            return forward(output, target);
        }
    };

    class MTENSOR_API CrossEntropyLoss: public Loss{

    public: 

        CrossEntropyLoss();
        Tensor forward(Tensor output, Tensor target) override;

    };

    class MTENSOR_API BCEWithLogitsLoss: public Loss{

        std::string m_reduction;

    public: 

        BCEWithLogitsLoss(std::string reduction = "mean");
        Tensor forward(Tensor output, Tensor target) override;

    };

    class MTENSOR_API MSELoss: public Loss{

    public: 

        MSELoss();
        Tensor forward(Tensor output, Tensor target) override;

    };

    class MTENSOR_API L1Loss: public Loss{

    public: 

        L1Loss();
        Tensor forward(Tensor output, Tensor target) override;

    };

    class MTENSOR_API KLDivLoss: public Loss{

    public: 

        KLDivLoss();
        Tensor forward(Tensor output, Tensor target) override;

    };


namespace optimizer{

    class MTENSOR_API Optimizer{

    protected:

        std::vector<Tensor> m_parameters;
        ops::Linear linear;
        float m_lr;

    public :

        Optimizer(const std::vector<Tensor>& parameters, float lr);
        virtual void step() = 0;
        std::vector<Tensor>& paramters();
        float get_lr() const;
        void set_lr(float lr);
        void zero_grad();
        
    };

    class MTENSOR_API Adam : public Optimizer{

        float m_beta1;
        float m_beta2;
        float m_beta1_t = 1.0f;
        float m_beta2_t = 1.0f;
        float m_eps;
        std::vector<Tensor> m;
        std::vector<Tensor> v;

    public: 
        
        Adam(
            const std::vector<Tensor>& parameters,
            float lr =  1e-3f, 
            float beta1 = 0.9,
            float beta2 = 0.999,
            float eps = 1e-8f 
        );

        void step() override;

        std::vector<Tensor>& stats_m();
        std::vector<Tensor>& stats_v();

    };


    class MTENSOR_API AdamW : public Optimizer{

        float m_beta1;
        float m_beta2;
        float m_beta1_t = 1.0f;
        float m_beta2_t = 1.0f;
        float m_eps;
        float m_weight_decay;
        std::vector<Tensor> m;
        std::vector<Tensor> v;

    public: 
        
        AdamW(
            const std::vector<Tensor>& parameters,
            float lr =  1e-3f, 
            float weight_decay = 0.01,
            float beta1 = 0.9,
            float beta2 = 0.999,
            float eps = 1e-8f 
        );

        void step() override;

        std::vector<Tensor>& stats_m();
        std::vector<Tensor>& stats_v();

    };

    class MTENSOR_API SGD : public Optimizer{

        float m_momentum;
        float m_weight_decay;
        std::vector<Tensor> v;

    public: 
        
        SGD(
            const std::vector<Tensor>& parameters,
            float lr =  1e-3f, 
            float weight_decay = 0.0,
            float momentum = 0.0
        );

        void step() override;

        std::vector<Tensor>& stats_v();

    };

}



    /************************************************* */
    //                    TRAINER                      // 
    /************************************************* */
    

    struct trainer_state{
        int64_t epoch;
        int64_t global_step;
        int64_t step_in_epoch; //step in the current epoch
        float train_loss; //average train loss for a complete epoch
        float train_accuracy;
        float val_loss;
        float val_accuracy;
        float last_train_loss; //loss of the last batch 
        bool stop_training; 
    };

    struct save_states{
        int64_t epoch;
        int64_t global_step;
        int64_t step_in_epoch; 
        float train_loss;
        float train_accuracy;
        float val_loss;
        float val_accuracy;
        float last_train_loss;
    };

    MTENSOR_API void save_model(Module& model,save_states& states, const std::string& path);
    MTENSOR_API save_states load_model(Module& model, const std::string& path);


    /****************************** */
    // this part is postponed to another round cause i dont have time for now.
    /****************************** */

    // class MTENSOR_API Callback{
    // public: 
    //     virtual void on_train_begin(const trainer_state& trainer_state) {};
    //     virtual void on_train_end(const trainer_state& trainer_state) {};
    //     virtual void on_epoch_begin(const trainer_state& trainer_state) {};
    //     virtual void on_epoch_end(const trainer_state& trainer_state) {};
    //     virtual void on_batch_begin(const trainer_state& trainer_state) {};
    //     virtual void on_batch_end(const trainer_state& trainer_state) {};
    // };


    // class MTENSOR_API ModelCheckpoint: public Callback{

    //     std::string file_path;
    //     std::string monitor;
    //     float best_metric;
    // public:
        
    //     ModelCheckpoint(
    //         const std::string& _file_path, 
    //         const std::string& _monitor,
    //         const std::string& mode
    //     );

    //     void on_epoch_end() override;

    // };


    // class MTENSOR_API Trainer{

    // public:
    //     Trainer(
    //         Module* _model,
    //         optimizer::Optimizer* _optimizer,
    //         std::function<float(std::vector<Tensor>,Tensor)> _loss_fn, 
    //         int64_t batch_size,
    //         data::DataLoader* _train_loader,
    //         std::vector<Callback*> _callbacks = {},
    //         data::DataLoader* _val_loader = nullptr,
    //         trainer_state _states = {}
    //     );

    //     void fit(int64_t num_epochs);

    //     Module* model;
    //     optimizer::Optimizer* optimizer;
    //     std::function<float(std::vector<Tensor>,Tensor)> loss_fn;
    //     data::DataLoader* train_loader;
    //     data::DataLoader* val_loader;
    //     std::vector<Callback*> callbacks;
    //     trainer_state states;
    //     int64_t batch_size;

    // private:

    //     void _trigger_callbacks(const std::string& event);
    //     void _train_epoch();
    //     void _validate_epoch();
    //     float _train_step(const std::pair<Tensor,Tensor>& batch);
    //     float _eval_step(const std::pair<Tensor,Tensor>& batch);

    // };


}//nn

namespace init{


    float calculate_gain(std::string nonlinearity, float a = 0);

    struct kaiming_uniform_{

        float a;  
        std::string nonlinearity;
        int64_t fan_mode;

        kaiming_uniform_(int64_t _fan_mode, std::string _nonlinearity = "leaky_relu", float _a = 0.0f):
        fan_mode(_fan_mode), nonlinearity(_nonlinearity), a(_a){}

        Tensor operator()(const Shape& shape, int64_t seed  = -1){
            float gain = calculate_gain(nonlinearity, a);
            float bound = gain * std::sqrt(3.0f / fan_mode);
            return Tensor::rand(shape, -bound, bound, seed);
        }
    };


    struct kaiming_normal_{

        float a;  
        std::string nonlinearity;
        int64_t fan_mode;

        kaiming_normal_(int64_t _fan_mode, std::string _nonlinearity = "leaky_relu", float _a = 0):
        fan_mode(_fan_mode), nonlinearity(_nonlinearity), a(_a){}

        Tensor operator()(const Shape& shape, int64_t seed = -1){
            float gain = calculate_gain(nonlinearity, a);
            float std = gain / std::sqrt(fan_mode);
            return Tensor::randn(shape, 0.0f, std , seed);
        }

    };

    struct xavier_uniform_{

        float a;  
        std::string nonlinearity;
        int64_t fan_in;
        int64_t fan_out;

        xavier_uniform_(int64_t _fan_in, int64_t _fan_out, std::string _nonlinearity = "sigmoid", float _a = 0):
        fan_in(_fan_in), fan_out(_fan_out), nonlinearity(_nonlinearity), a(_a){}

        Tensor operator()(const Shape& shape, int64_t seed = -1){
            float gain = calculate_gain(nonlinearity);
            float bound = gain * std::sqrt( 6.0f / (fan_in + fan_out) );
            return Tensor::rand(shape, -bound , bound, seed);
        }

    };

    
    struct xavier_normal_{

        float a;  
        std::string nonlinearity;
        int64_t fan_in;
        int64_t fan_out;

        xavier_normal_(int64_t _fan_in, int64_t _fan_out, std::string _nonlinearity = "sigmoid", float _a = 0):
        fan_in(_fan_in), fan_out(_fan_out), nonlinearity(_nonlinearity), a(_a){}

        Tensor operator()(const Shape& shape, int64_t seed = -1){
            float gain = calculate_gain(nonlinearity);
            float std = gain * std::sqrt( 2.0f / (fan_in + fan_out) );
            return Tensor::randn(shape, 0.0f, std , seed);
        }

    };

}//init  

}//mt


#endif //NN_HPP

