﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace KelpNet
{
    //FunctionStackに積み上げるFunctionの基底クラス
    [Serializable]
    public abstract class Function : IFunction
    {
        public string Name { get; set; }

        public NdArray[] Parameters { get; set; }

        public Optimizer[] Optimizers { get; set; }

        [field: NonSerialized]
        public List<NdArray[]> PrevInputs { get; set; }

        [field: NonSerialized]
        public List<NdArray[]> UsedPrevInputs { get; set; }

        public abstract NdArray[] Forward(params NdArray[] xs);
        public abstract NdArray[] Predict(params NdArray[] xs);
        public virtual void Backward(params NdArray[] ys) { }

        public string[] InputNames { get; set; }
        public string[] OutputNames { get; set; }

        //コンストラクタ
        protected Function(string name, string[] inputNames = null, string[] outputNames = null)
        {
            this.Name = name;

            if (inputNames != null)
            {
                this.InputNames = inputNames.ToArray();
            }

            if (outputNames != null)
            {
                this.OutputNames = outputNames.ToArray();
            }

            this.PrevInputs = new List<NdArray[]>();
            this.UsedPrevInputs = new List<NdArray[]>();

            Parameters = new NdArray[] { };
            Optimizers = new Optimizer[] { };
        }

        public virtual void SetOptimizer(params Optimizer[] optimizers)
        {
            this.Optimizers = optimizers;

            foreach (Optimizer optimizer in optimizers)
            {
                optimizer.AddFunctionParameters(this.Parameters);
            }
        }

        //パラメータを更新する時に呼ぶ関数
        protected void BackwardCountUp()
        {
            foreach (NdArray parameter in this.Parameters)
            {
                parameter.CountUp();
            }
        }

        public void InitGrad()
        {
            foreach (NdArray parameter in this.Parameters)
            {
                if (parameter.Grad == null) parameter.InitGrad();
            }
        }

        public virtual void Update()
        {
            foreach (Optimizer optimizer in this.Optimizers)
            {
                optimizer.Update();
            }
        }

        //RNN等で使い切れなかった入力データを初期化
        public virtual void ResetState()
        {
            this.PrevInputs = new List<NdArray[]>();
            this.UsedPrevInputs = new List<NdArray[]>();
        }

        //名前を返す
        public override string ToString()
        {
            return this.Name;
        }
    }
}
