/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/


#ifndef ERRORFUNCTIONS_H_
#define ERRORFUNCTIONS_H_

#define ERROR_LINEAR 0
#define ERROR_TANH 1

//returns the new error after the application of a function (tanh is more aggressive error targeting)
inline float errorFunction(float error, int func){
	switch(func){
			case ERROR_TANH:	if		(error < -.9999999)			return -17.0;
								else if	(error >  .9999999)			return 17.0;
								else 								return log((1.0 + error) / (1.0 - error));
			case ERROR_LINEAR:	return error;
			default:			printf("FUNCTION NOT IMPLEMENTED YET\n");exit(1);
		}
}


#endif /* ERRORFUNCTIONS_H_ */
