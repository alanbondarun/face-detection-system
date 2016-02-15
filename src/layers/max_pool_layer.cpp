#include "layers/max_pool_layer.hpp"
#include "calc/calc-cpu.hpp"
#include "utils/make_unique.hpp"

namespace NeuralNet
{
	MaxPoolLayer::MaxPoolLayer(const Dimension& dim)
		: m_dim(dim), m_output_width(m_dim.image_width / m_dim.pool_width),
		m_output_height(m_dim.image_height / m_dim.pool_height)
	{
	}
	
	void MaxPoolLayer::forward_cpu(const LayerData& prev, LayerData& current)
	{
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto prev_z = prev.get(LayerData::DataIndex::INTER_VALUE);
		auto cur_a = current.get(LayerData::DataIndex::ACTIVATION);
		auto cur_z = current.get(LayerData::DataIndex::INTER_VALUE);
		
		for (size_t i=0; i<m_dim.train_num; i++)
		{
			for (size_t j=0; j<m_dim.map_num; j++)
			{
				size_t back_offset = sizeof(double) * (i * m_dim.map_num + j)
						* (m_dim.image_width * m_dim.image_height);
				size_t front_offset = sizeof(double) * (i * m_dim.map_num + j)
						* (m_output_width * m_output_height);
				downsample_max(prev_z + back_offset, cur_z + front_offset,
						m_dim.image_width, m_dim.image_height,
						m_dim.pool_width, m_dim.pool_height);
				downsample_max(prev_a + back_offset, cur_a + front_offset,
						m_dim.image_width, m_dim.image_height,
						m_dim.pool_width, m_dim.pool_height);
			}
		}
	}
	
	void MaxPoolLayer::forward_gpu(const LayerData& prev, LayerData& current)
	{
		/* TODO: OpenCL intergration */
	}
	
	void MaxPoolLayer::backward_cpu(LayerData& prev, LayerData& current)
	{
		auto prev_e = prev.get(LayerData::DataIndex::ERROR);
		auto prev_a = prev.get(LayerData::DataIndex::ACTIVATION);
		auto cur_e = current.get(LayerData::DataIndex::ERROR);
		
		for (size_t i=0; i<m_dim.train_num; i++)
		{
			for (size_t j=0; j<m_dim.map_num; j++)
			{
				size_t back_offset = sizeof(double) * (i * m_dim.map_num + j)
						* (m_dim.image_width * m_dim.image_height);
				size_t front_offset = sizeof(double) * (i * m_dim.map_num + j)
						* (m_output_width * m_output_height);
						
				upsample_max(cur_e + front_offset, prev_a + back_offset,
						prev_e + back_offset,
						m_dim.image_width, m_dim.image_height,
						m_dim.pool_width, m_dim.pool_height);
			}
		}
	}
	
	void MaxPoolLayer::backward_gpu(LayerData& prev, LayerData& current)
	{
		/* TODO: OpenCL intergration */
	}
	
	std::unique_ptr<LayerData> MaxPoolLayer::createLayerData()
	{
		return std::make_unique<LayerData>(
			m_dim.train_num,
			m_dim.map_num * m_output_width * m_output_height
		);
	}
}