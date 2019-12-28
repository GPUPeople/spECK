#pragma once

#include "dCSR.h"

namespace spECK {
	template <typename DataType>
	bool Compare(const dCSR<DataType>& reference_mat, const dCSR<DataType>& compare_mat, bool compare_data);
}