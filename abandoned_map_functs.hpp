struct conv2d_jacobi_map1 : map_base
    {
        // parent节点为待卷积的节点
        conv2d_jacobi_map1(vector<int>& dvp, vector<int>& dvc)
        {
            this->dsize_p = dvp;
            this->dsize_c = dvc;
        }
        vector<int> operator()(int x, int y)
        {
            // J(oc * (m - r + 1) * (n - s + 1) + i * (n - s + 1) + j, ic * m * n + p * n + q)
            //             = kernel(oc, ic, p - i, q - j);
            // dvp = {ics, m, n}
            // dvc = {ocs, m-r+1, n-s+1}
            auto& dvp = this->dsize_p;
            auto& dvc = this->dsize_c;
            // cout << "extra_nodes.hpp: op1 called. " << dvp << ' ' << dvc << endl;

            int oc = x / dvc[1] / dvc[2];
            int ic = y / dvp[1] / dvp[2];
            int mod1 = x % (dvc[1] * dvc[2]);
            int i = mod1 / dvc[2];
            int j = mod1 % dvc[2];
            int mod2 = y % (dvp[1] * dvp[2]);
            int p = mod2 / dvp[2];
            int q = mod2 % dvp[2];
            return {oc, ic, p - i, q - j};
        }
    };

    struct conv2d_jacobi_map2 : map_base
    {
        // parent节点为卷积核
        conv2d_jacobi_map2(vector<int>& dvp, vector<int>& dvc)
        {
            this->dsize_p = dvp;
            this->dsize_c = dvc;
        }
        vector<int> operator()(int x, int y)
        {
            // J(oc * (m - r + 1) * (n - s + 1) + i * (n - s + 1) + j, 
            //             oc * this->in_channels * r * s + ic * r * s + u * s + v) 
            //         = data_in(ic, i + u, j + v);
            // dvp = {ocs, ics, r, s}
            // dvc = {ocs, m-r+1, n-s+1}
            auto& dvp = this->dsize_p;
            auto& dvc = this->dsize_c;
            // cout << "extra_nodes.hpp: op2 called. " << dvp << ' ' << dvc << endl;

            int mod1 = x % (dvc[1] * dvc[2]);
            int i = mod1 / dvc[2];
            int j = mod1 % dvc[2];
            int mod2 = y % (dvp[1] * dvp[2] * dvp[3]);
            int ic = mod2 / (dvp[2] * dvp[3]);
            int mod3 = mod2 % (dvp[2] * dvp[3]);
            int u = mod3 / dvp[3];
            int v = mod3 % dvp[3];
            return {ic, i + u, j + v};
        }
    };

    // map_base* map = new conv2d_jacobi_map1(data_in.dsize, this->value->dsize);
                // cout << "exnodes: " << map->dsize_p << ' ' << map->dsize_c << endl;
                // basic_tensor<T> J(kernel.data, kernel.dsize, default_TensorOptions, true, map,
                //                     {this->value->data_size(), this->parents[0]->value->data_size()});
                // cout << "extra_nodes.hpp: " << this->value->dsize << ' ' << J.use_mirror << ' ' << J.dsize << ' ' << 
                //             J.map_dsize << ' ' << "flag1.\n";

   // map_base* map = new conv2d_jacobi_map2(this->parents[1]->value->dsize, this->value->dsize);
                // cout << "exnodes: " << map->dsize_p << ' ' << map->dsize_c << endl;
                // basic_tensor<T> J(data_in.data, data_in.dsize, default_TensorOptions, true, map,
                //                     {this->value->data_size(), this->parents[1]->value->data_size()});
                // cout << "extra_nodes.hpp: " << this->value->dsize << ' ' << J.use_mirror << ' ' << J.dsize << ' ' 
                //          << J.map_dsize << " flag2.\n";

            struct map_base
    {
        vector<int> dsize_din;
        vector<int> dsize_w;
        vector<int> dsize_dout;
        virtual int operator()(int ,int) = 0;
    };

            // bool use_mirror;  // 不实际存储所有元素，只是存储一部分，访问时用映射对应相应元素(得不偿失，耗时过多，不如用空间换时间)
        // map_base* data_map;  // 映射函数
        // vector<int> map_dsize;