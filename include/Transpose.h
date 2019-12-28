#pragma once

#include "dCSR.h"

namespace spECK {
	template <typename DataType>
	void Transpose(const dCSR<DataType>& matIn, dCSR<DataType>& matTransposeOut);
}