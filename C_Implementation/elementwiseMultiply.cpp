
#include "elementwiseMultiply.h"

#include <iostream>


void elementwiseMultiply(
    const float matrixIn1[IMAGE_ROWS][IMAGE_COLS],
    const float matrixIn2[IMAGE_ROWS][IMAGE_COLS],
    float matrixOut[IMAGE_ROWS][IMAGE_COLS]
) {
    for (int i = 0; i < IMAGE_ROWS; i++) {
        for (int j = 0; j < IMAGE_COLS; j++) {
            matrixOut[i][j] = matrixIn1[i][j] * matrixIn2[i][j];
        }
    }
}

void displayMatrix(const float matrix[IMAGE_ROWS][IMAGE_COLS]) {
    for (int i = 0; i < IMAGE_ROWS; i++) {
        for (int j = 0; j < IMAGE_COLS; j++) {
            std::cout << matrix[i][j] << " \t";
        }
        std::cout << std::endl;   
    }
    
}


// int main() {

//     float matrixAns[IMAGE_ROWS][IMAGE_COLS] = {0};

//     std::cout << "Matrix A:" << std::endl;
//     displayMatrix(matrixA);

//     std::cout << "\nMatrix B:" << std::endl;
//     displayMatrix(matrixB);

//     std::cout << "\nHadamard Product::" << std::endl;
//     displayMatrix(hadamardProduct);

//     elementwiseMultiply(matrixA, matrixB, matrixAns);
//     std::cout << "\nMatrix Answer:" << std::endl;
//     displayMatrix(matrixAns);

//     return 0;
// }
