#include "SchismGeometry.h"
#include "MeshConstants.h"


int maxNodeInCell = MeshConstants::MAX_NUM_NODE_PER_CELL;

int local3SideNode[3][2]={{1,2},
                          {2,0},
                          {0,1}};

int local4SideNode[4][2]={{1,2},
                          {2,3},
                          {3,0},
                          {0,1}};


int meshSideNum( int    *     a_meshNode,
                 const int &  a_numberEle,
				 const int &  a_numberNode)
{
	int * numEleAtNode = new int [a_numberNode];
	for(int i=0;i<a_numberNode;i++)
	{
		numEleAtNode[i]=0;
	}

    for(int iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(maxNodeInCell+1)];
        for (int iNode=0;iNode<numNode;iNode++)
		{
          int node=a_meshNode[iele*(maxNodeInCell+1)+1+iNode]-1;
          numEleAtNode[node]+=1;
		}
	}
    
	int maxEleAtNode = 0;

	for(int i=0;i<a_numberNode;i++)
	{
		if (numEleAtNode[i]>maxEleAtNode)
		{
			maxEleAtNode = numEleAtNode[i];
		}
	}

	int * eleAtNode = new int  [maxEleAtNode*a_numberNode];

	for(int i=0;i<maxEleAtNode*a_numberNode;i++)
	{
		eleAtNode[i]=-1;
	}

	for(int i=0;i<a_numberNode;i++)
	{
		numEleAtNode[i]=0;
	}

	for(int iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(maxNodeInCell+1)];
        for (int iNode=0;iNode<numNode;iNode++)
		{
          int node=a_meshNode[iele*(maxNodeInCell+1)+1+iNode]-1;
		  int  index = node*maxEleAtNode+numEleAtNode[node];
		  eleAtNode[index]=iele;
          numEleAtNode[node]+=1;
		}
	}

	int * neiborEleAllSide = new int [maxNodeInCell*a_numberEle];
	for(int i=0;i<maxNodeInCell*a_numberEle;i++)
	{
		neiborEleAllSide[i]=-1;
	}
	for(int iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(maxNodeInCell+1)];
		int numSide = numNode;
        for (int iSide=0;iSide<numSide;iSide++)
		{
		  int node1=-1;
		  int node2=-1;
		  int idStart = iele*(maxNodeInCell+1)+1;
          if(numSide==3)
		  {
			  node1= a_meshNode[idStart+local3SideNode[iSide][0]]-1;
			  node2= a_meshNode[idStart+local3SideNode[iSide][1]]-1;
		  }
		  else
		  {
			  node1= a_meshNode[idStart+local4SideNode[iSide][0]]-1;
			  node2= a_meshNode[idStart+local4SideNode[iSide][1]]-1;
		  }

		  for(int iNeiborEle=0;iNeiborEle<numEleAtNode[node1];iNeiborEle++)
		  {
			  int eleIndex = eleAtNode[node1*maxEleAtNode+iNeiborEle];
			  if ((eleIndex>-1)&&(eleIndex != iele))
			  {
				  int eleStartIndex = eleIndex*(maxNodeInCell+1);
				  int numNode = a_meshNode[eleStartIndex];
				  bool shareThisSide = false;
				  for(int iiNode=0;iiNode<numNode;iiNode++)
				  {
					  int nodeIndex = a_meshNode[eleStartIndex+1+iiNode]-1;
					  if (nodeIndex==node2)
					  {
						  shareThisSide = true;
						  break;
					  }

				  }
				  if (shareThisSide)
				  {
					  neiborEleAllSide[maxNodeInCell*iele+iSide]=eleIndex;
				  }
			  }
		  }
		}
	}

	int totalNumSide =0;

	for(int iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(maxNodeInCell+1)];
		int numSide = numNode;
        for (int iSide=0;iSide<numSide;iSide++)
		{
			int neiborEle = neiborEleAllSide[maxNodeInCell*iele+iSide];
			if ((neiborEle == -1) || (iele<neiborEle))
			{
				totalNumSide++;
			}

		}

	}
	delete numEleAtNode;
	delete eleAtNode;
	delete neiborEleAllSide;
	return totalNumSide;
}

void   meshSideNode(int    *     a_sideNode,
				    int    *     a_meshNode,
				    const int &  a_numberSide,
                    const int &  a_numberEle,
				    const int &  a_numberNode)
{
	int * numEleAtNode = new int [a_numberNode];
	for(int i=0;i<a_numberNode;i++)
	{
		numEleAtNode[i]=0;
	}

    for(int iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(maxNodeInCell+1)];
        for (int iNode=0;iNode<numNode;iNode++)
		{
          int node=a_meshNode[iele*(maxNodeInCell+1)+1+iNode]-1;
          numEleAtNode[node]+=1;
		}
	}
    
	int maxEleAtNode = 0;

	for(int i=0;i<a_numberNode;i++)
	{
		if (numEleAtNode[i]>maxEleAtNode)
		{
			maxEleAtNode = numEleAtNode[i];
		}
	}

	int * eleAtNode = new int  [maxEleAtNode*a_numberNode];

	for(int i=0;i<maxEleAtNode*a_numberNode;i++)
	{
		eleAtNode[i]=-1;
	}

	for(int i=0;i<a_numberNode;i++)
	{
		numEleAtNode[i]=0;
	}

	for(int iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(maxNodeInCell+1)];
        for (int iNode=0;iNode<numNode;iNode++)
		{
          int node=a_meshNode[iele*(maxNodeInCell+1)+1+iNode]-1;
		  int  index = node*maxEleAtNode+numEleAtNode[node];
		  eleAtNode[index]=iele;
          numEleAtNode[node]+=1;
		}
	}

	int * neiborEleAllSide = new int [maxNodeInCell*a_numberEle];
	for(int i=0;i<maxNodeInCell*a_numberEle;i++)
	{
		neiborEleAllSide[i]=-1;
	}
	for(int iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(maxNodeInCell+1)];
		int numSide = numNode;
        for (int iSide=0;iSide<numSide;iSide++)
		{
		  int node1=-1;
		  int node2=-1;
		  int idStart = iele*(maxNodeInCell+1)+1;
          if(numSide==3)
		  {
			  node1= a_meshNode[idStart+local3SideNode[iSide][0]]-1;
			  node2= a_meshNode[idStart+local3SideNode[iSide][1]]-1;
		  }
		  else
		  {
			  node1= a_meshNode[idStart+local4SideNode[iSide][0]]-1;
			  node2= a_meshNode[idStart+local4SideNode[iSide][1]]-1;
		  }

		  for(int iNeiborEle=0;iNeiborEle<numEleAtNode[node1];iNeiborEle++)
		  {
			  int eleIndex = eleAtNode[node1*maxEleAtNode+iNeiborEle];
			  if((eleIndex>-1)&&(eleIndex != iele))
			  {
				  int eleStartIndex = eleIndex*(maxNodeInCell+1);
				  int numNode = a_meshNode[eleStartIndex];
				  bool shareThisSide = false;
				  for(int iiNode=0;iiNode<numNode;iiNode++)
				  {
					  int nodeIndex = a_meshNode[eleStartIndex+1+iiNode]-1;
					  if (nodeIndex==node2)
					  {
						  shareThisSide = true;
						  break;
					  }

				  }
				  if (shareThisSide)
				  {
					  neiborEleAllSide[maxNodeInCell*iele+iSide]=eleIndex;
				  }
			  }
		  }
		}
	}

	
	

	int totalNumSideTemp =0;
	
	for(int iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(maxNodeInCell+1)];
		int numSide = numNode;
        for (int iSide=0;iSide<numSide;iSide++)
		{
			int node1=-1;
		    int node2=-1;
		    int idStart = iele*(maxNodeInCell+1)+1;
            if(numSide==3)
		    {
			    node1= a_meshNode[idStart+local3SideNode[iSide][0]]-1;
			    node2= a_meshNode[idStart+local3SideNode[iSide][1]]-1;
		    }
		    else
		    {
			    node1= a_meshNode[idStart+local4SideNode[iSide][0]]-1;
			    node2= a_meshNode[idStart+local4SideNode[iSide][1]]-1;
		    }
			int neiborEle = neiborEleAllSide[maxNodeInCell*iele+iSide];
			if ((neiborEle == -1) || (iele<neiborEle)) // a new side
			{
				a_sideNode[2*totalNumSideTemp]  =node1;
				a_sideNode[2*totalNumSideTemp+1]=node2;
				totalNumSideTemp++;
			}
		}
	}

	

	delete numEleAtNode;
	delete eleAtNode;
	delete neiborEleAllSide;



}