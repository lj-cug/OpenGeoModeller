#ifndef _SCHISMGEOMETRY_H_
#define _SCHISMGEOMETRY_H_


int  meshSideNum( int    *     a_meshNode,
                  const int &  a_numberEle,
				  const int &  a_numberNode);


void   meshSideNode(int    *     a_sideNode,
				    int    *     a_meshNode,
				    const int &  a_numberSide,
                    const int &  a_numberEle,
				    const int &  a_numberNode);

#endif