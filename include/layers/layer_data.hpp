#ifndef __LAYER_DATA_HPP
#define __LAYER_DATA_HPP

namespace NeuralNet
{
    class LayerData
    {
    public:
        enum class DataIndex
        {
            ACTIVATION,
            INTER_VALUE,
            WEIGHT,
            BIAS,
			ERROR,
            START = ACTIVATION,
            END = ERROR
        };

        LayerData(size_t prev_len, size_t current_len);
		~LayerData();
		
		/* resize the arrays */
		void resize(size_t prev_len, size_t current_len);
		
		/* returns the desired array */
		double *get(DataIndex idx) const;

    private:
        size_t m_prev_len, m_current_len;
        double **data;
		constexpr size_t DATA_COUNT = static_cast<int>(DataIndex::END)
				- static_cast<int>(DataIndex::START) + 1;
    };
}

#endif // __LAYER_DATA_HPP
