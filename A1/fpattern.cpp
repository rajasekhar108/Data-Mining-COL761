//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <bits/stdc++.h>
#include<map>
#include<cmath>
#include <algorithm>
#include<functional>
#include<set>
#include<time.h>
#include<stdlib.h>
#include<chrono>
#include<limits.h>
using namespace std;
ofstream outfile_sol;
string in_file_name;
string outfile_name;
int total_items_in_file=0;
#include <memory>
#include<vector>

using namespace std;
#ifndef UNTITLED_STRUC_DEF_H
#define UNTITLED_STRUC_DEF_H

#endif 

#include <map>
#include <vector>
#include <algorithm>

class Bucket{
public:
    int numb;
    int numbC;
    shared_ptr<Bucket> next_same_numb_ptr; 
    unordered_map<int,shared_ptr<Bucket>> child_bucket_dict;
   shared_ptr<Bucket> parent_bucket_ptr;

 Bucket(int numb, shared_ptr<Bucket> parent_bucket_ptr, int numbC=1) {
     this->numb = numb;
     this->numbC = numbC
             ;
     this->parent_bucket_ptr = parent_bucket_ptr;
        this->next_same_numb_ptr=NULL;
 }

};


typedef std::pair<int,int> pair_;

class Tree{
public:

    unordered_map<int,shared_ptr<Bucket>> dictionary_first_bucket;
    shared_ptr<Bucket> tree_root;
    vector<pair_> vec_freq_num_sorted_dec;
};
using namespace std::chrono;
auto start = high_resolution_clock::now();

int get_time_spent_seconds(){
auto stop = high_resolution_clock::now();
auto duration = duration_cast<seconds>(stop-start);
return duration.count();

}

int nextint_fastio(FILE*fs, int &number )
{

    number = INT_MIN;
    int alphabet = getc_unlocked(fs);
    if(alphabet==EOF)
        return 0; 

    bool neg = false;

    if(alphabet=='-'){
        neg =true;
        alphabet = getc_unlocked(fs);
    }
    while((alphabet > 47 && alphabet < 58)){
	if( number !=INT_MIN){
        number = number *10 + alphabet - 48;
	}
	else{
		number = alphabet-48;	
	}

        alphabet=getc_unlocked(fs);
    }
    if(neg)
        number = number*-1;
    if(alphabet == '\n')
        return 1; 
    return 2; 
}


int fastscan_first_pass(FILE *fs, unordered_map<int,int> &numbC_dict, int &num_transactions) {
    register int number;

    register int flag = 2;

    while(flag){
        flag = nextint_fastio(fs,number);
        if(number != INT_MIN){
            numbC_dict[number] += 1;
            total_items_in_file++;
        }
        if((flag ==1 || flag == 0) ){
            num_transactions += 1;
            if (num_transactions%10000 ==0) {
            }
        }
    }
}


int fastscan_second_pass(FILE *fs,vector<int>& transaction, unordered_map<int,int> &dict_filtered)
{


    register int number;
    register int flag = 2;

    while(flag==2){
        flag = nextint_fastio(fs,number);
        if(number != INT_MIN)
        {
            if( dict_filtered.find(number)!= dict_filtered.end()){
                transaction.emplace_back(number);
            }
        }

    }

    return flag;

}





// create frequecy dictionary
unordered_map<int, int>  get_Freq_From_File(string filePath, int &num_of_transactions){

    unordered_map<int, int> numbC_dict;
    FILE* inFile = fopen(filePath.c_str(), "r");
    fastscan_first_pass(inFile, numbC_dict, num_of_transactions);

    return numbC_dict;
}

// function to remove non frequenct items.
unordered_map<int, int> filter_numbs_using_support( unordered_map<int, int> &numbC_dict , int support_count){
    unordered_map<int, int> numbC_dict_filtered ;
//   numbC_dict_filtered.reserve(5000);
    for (unordered_map<int,int>::iterator it=numbC_dict.begin(); it!=numbC_dict.end(); ++it) 
    {
        if (it->second >= support_count){
            numbC_dict_filtered[it->first] = it->second;
        }
    }
    return numbC_dict_filtered;
}


void init_Tree(Tree& tree_obj){
    tree_obj.tree_root =make_shared<Bucket>(-1, nullptr);
    tree_obj.dictionary_first_bucket.clear();
//tree_obj.dictionary_most_recent_bucket.clear();

    tree_obj.vec_freq_num_sorted_dec.clear();
}

// check if the item exists in dictionary
bool exists_in_dict(int name,unordered_map<int, shared_ptr<Bucket> > dictionary )
{
    if ( dictionary.find(name) == dictionary.end() ) {
        return 0;
    } else {
        return 1;
    }
}
typedef std::pair<int,int> pair_;


//SECOND PAS
Tree create_cond_tree(Tree &tree_obj, vector<pair<vector<int>,int>> &list_ptrns_with_count ,unordered_map<int,int> &dict_count_numb_in_tree, int support_count)
{
    init_Tree(tree_obj);
    if( list_ptrns_with_count.size() ==0)
        return tree_obj;
// filtered dictionary for new tree
    unordered_map<int,int>   dict_count_numb_in_tree_filtered= filter_numbs_using_support(dict_count_numb_in_tree, support_count);

    for ( vector < pair<vector<int>,int>>::const_iterator it = list_ptrns_with_count.begin() ; it != list_ptrns_with_count.end(); it++){

        auto ptr = tree_obj.tree_root;

          auto filter = [&](int numb) -> bool {
               return dict_count_numb_in_tree_filtered.find(numb)== dict_count_numb_in_tree_filtered.end();
            };

        vector<int>ptrns = it->first;


	ptrns.erase(remove_if(ptrns.begin(), ptrns.end(), filter ) , ptrns.end()); //removing less than support ones
	//cout<<"count after "<< ptrns.size()<<endl;	
        int freq_of_numb = it->second;
        auto comp = [&](int numb1, int numb2) -> bool {
            int numb_1_freq = 0;
            int numb_2_freq = 0;
     

            return dict_count_numb_in_tree_filtered[numb1] > dict_count_numb_in_tree_filtered[numb2];
        };	
        // sort transaction

	  sort(ptrns.begin(), ptrns.end(), comp);
	///cout<<"count after "<< dict_count_numb_in_tree_filtered.size();
        for (vector<int>::const_iterator it = ptrns.begin();it != ptrns.end(); it++ ){
            int numb = *it;
           
			// insert elements in tree
		{
               

                if (ptr->child_bucket_dict.find(numb)!=ptr->child_bucket_dict.end()){//exists_in_dict(numb,ptr->child_bucket_dict)){
                    shared_ptr<Bucket> child_ptr = ptr->child_bucket_dict[numb];
                    (*child_ptr).numbC+= freq_of_numb;
                    ptr = child_ptr;

                }
                else{

                    //   cout<<endl<<" could not find "<<numb;
                    shared_ptr<Bucket>  child_node_ptr  (new Bucket(numb,ptr, freq_of_numb));
                    ptr->child_bucket_dict[numb] = child_node_ptr;
	
                    if(tree_obj.dictionary_first_bucket.find(numb)!= tree_obj.dictionary_first_bucket.end())//  (exists_in_dict(numb,tree_obj.dictionary_first_bucket))
                    {

                        child_node_ptr->next_same_numb_ptr = tree_obj.dictionary_first_bucket[numb];
                        tree_obj.dictionary_first_bucket[numb] = child_node_ptr;
                    }
                    else{
                        tree_obj.dictionary_first_bucket[numb] = child_node_ptr;

                    }

                    ptr = child_node_ptr;
                }
            }
//            else
		{
                //        break;
            }

        }
    }

    std::vector<pair_> vec_freq_num_sorted_dec;
    std::copy(dict_count_numb_in_tree_filtered.begin(),
              dict_count_numb_in_tree_filtered.end(),
              std::back_inserter<std::vector<pair_>>(vec_freq_num_sorted_dec));
    std::sort(vec_freq_num_sorted_dec.begin(), vec_freq_num_sorted_dec.end(),
              [](const pair_& l, const pair_& r) {
                  if (l.second != r.second)
                      return l.second < r.second;

                  return l.first < r.first;
              });
    tree_obj.vec_freq_num_sorted_dec = vec_freq_num_sorted_dec;

    return tree_obj;
}


typedef std::pair<int,int> pair_;

//start mining patterns
void pattern_mine(Tree tree_obj, int support_count, vector<string> mined_part, int clock_counter=0){
    Tree tree_obj_new;

    vector<pair<vector<int>,int>> list_ptrns_with_count;


	int clock_counter_2 = 0 ;
    for (auto const &pair_iter: tree_obj.vec_freq_num_sorted_dec) {
        unordered_map<int,int> dict_count_numb_in_tree;
	clock_counter_2++;

        list_ptrns_with_count.clear();
        vector<string> mined_part_2(mined_part.begin(), mined_part.end());
        mined_part_2.emplace_back(to_string(pair_iter.first));
       // cout << "cnd " << pair_iter.first;
       // std::cout << '{' << pair_iter.first << "," << pair_iter.second << '}' << '\n';
        shared_ptr<Bucket>  ptr = tree_obj.dictionary_first_bucket[pair_iter.first];
        int freq_of_items_prefix_path = 0;
        while (ptr) {

            vector<int> ptrns;

            shared_ptr<Bucket>  parent_ptr = ptr->parent_bucket_ptr;
            freq_of_items_prefix_path = ptr->numbC;


            while (parent_ptr && parent_ptr->parent_bucket_ptr) {

                ptrns.emplace_back(parent_ptr->numb);

                dict_count_numb_in_tree[parent_ptr->numb]+=freq_of_items_prefix_path;

                parent_ptr = parent_ptr->parent_bucket_ptr;

            }
        
            if (!ptrns.empty()){
              //  std::reverse(ptrns.begin(), ptrns.end());
                list_ptrns_with_count.emplace_back(make_pair(ptrns, freq_of_items_prefix_path));

            }

            ptr = ptr->next_same_numb_ptr;// do for everyone with same id. along the list [a] --->[a] -----[a]

         //   cout << "";
        }

	// create conditional tree

        tree_obj_new = create_cond_tree(tree_obj_new, list_ptrns_with_count, dict_count_numb_in_tree,support_count);

        if (tree_obj_new.dictionary_first_bucket.size() != 0)
        {
            if(list_ptrns_with_count.size()){
                pattern_mine(tree_obj_new, support_count, mined_part_2, clock_counter+1);

            }

        }
      //  cout<<endl;
        //cout<<"mnd "<<endl;
        std::sort(mined_part_2.begin(), mined_part_2.end());
        for (string x : mined_part_2) {
            outfile_sol <<x<<" ";

            //mcout << x << " ";
        }
	
        outfile_sol<<"\n";
        //mcout<<endl;
    }

}

// function to create tree from transactions
void create_Tree(Tree &tree_obj, unordered_map<int, int> filtered_numb_dict, string filePath) {
    //mcout << " Building tree";
    if (filtered_numb_dict.size() == 0) {
        return;

    }

    std::ifstream file(filePath);

    std::string line;
    auto ptr = tree_obj.tree_root;

    typedef std::pair<int, int> pair;

    std::vector<pair> vec_freq_num_sorted_dec;
    std::copy(filtered_numb_dict.begin(),
              filtered_numb_dict.end(),
              std::back_inserter<std::vector<pair>>(vec_freq_num_sorted_dec));
    std::sort(vec_freq_num_sorted_dec.begin(), vec_freq_num_sorted_dec.end(),
              [](const pair& l, const pair& r) {
                  if (l.second != r.second)
                      return l.second < r.second;

                  return l.first < r.first;
              });

    tree_obj.vec_freq_num_sorted_dec = vec_freq_num_sorted_dec;


    int transaction = 0;
    FILE *inFile = fopen(filePath.c_str(), "r");
    file.tie(NULL);
    vector<int> current_transaction;

    while (fastscan_second_pass(inFile, current_transaction, filtered_numb_dict)) {



        transaction++;

        if (!current_transaction.empty())
        {

            ptr = tree_obj.tree_root;
            auto comp = [&](int numb1, int numb2) -> bool {
                int numb_1_freq = filtered_numb_dict[numb1];
                int numb_2_freq = filtered_numb_dict[numb2];

                return numb_1_freq > numb_2_freq;
            };
            // sort transaction based on freq of items
            sort(current_transaction.begin(), current_transaction.end(), comp);

            for (auto numb : current_transaction) {

                int found = 0;

                auto it = ptr->child_bucket_dict.find(numb);
                if (it != ptr->child_bucket_dict.end()) {

                    (*(it->second)).numbC++;

                    ptr = it->second;


                } else {

                    shared_ptr<Bucket>  child_node_ptr(new Bucket(numb, ptr));
                    ptr->child_bucket_dict[numb] = child_node_ptr;

                    if (tree_obj.dictionary_first_bucket.find(numb) != tree_obj.dictionary_first_bucket.end()) {
                        child_node_ptr->next_same_numb_ptr = tree_obj.dictionary_first_bucket[numb];
                        tree_obj.dictionary_first_bucket[numb] = child_node_ptr;


                    } else {
                        tree_obj.dictionary_first_bucket[numb] = child_node_ptr;

                    }

                    ptr = child_node_ptr;
                }
            }
            current_transaction.clear();

        }

    }
}




int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(false);


    in_file_name = argv[1];// changed arg to 1 to input file
    //outfile_name = argv[3];
    outfile_name = "frq.txt"; //freq pattern file is created
    int num_of_transactions=0;

    unordered_map<int, int> numb_count = get_Freq_From_File(in_file_name, num_of_transactions);
    vector<int>temp;
    for(auto i:numb_count) temp.push_back(i.second);
    int ti=accumulate(temp.begin(), temp.end(), 0);
    int med=(numb_count.size()*(0.5));
    // cout<<"med : "<<med<<endl;
    sort(temp.begin(),temp.end());
    //for(auto i : temp) cout<<i<<endl;///remove this
    int support_count=temp[(temp.size()*0.5)];
    temp.clear();
    double support_percentage=(support_count/(0.01*num_of_transactions));
    // cout<<"before : "<<support_percentage<<endl;
    if(ti>50000000) {support_percentage=12.0;}
    else if((ti>10000000) && (ti<50000000)) {support_percentage=10.0;}
    else if((ti>6000000) && (ti<10000000)) {support_percentage=0.8;cout<<"inif : "<<support_percentage<<endl;}

    // cout<<"after : "<<support_percentage<<endl;
    support_count = ceil(0.01*support_percentage*num_of_transactions);//
    // cout<<"no of items : "<<ti<<endl;
    // cout<<"support_percentage : "<<support_percentage<<endl<<"support_count : "<<support_count<<endl;
  
    outfile_sol.open(outfile_name);

    unordered_map<int, int> filtered_numb_dict = filter_numbs_using_support(numb_count, support_count);

    if (filtered_numb_dict.size() <=1){

        if (filtered_numb_dict.size() ==1){
            outfile_sol<<(((filtered_numb_dict.begin())->first));
            

        } else
        {
            outfile_sol<<("");
        }

    }

    else{
        Tree tree_obj ;
        init_Tree(tree_obj);
       //m cout<<" level 1 "<< filtered_numb_dict.size()<<endl;

        create_Tree(tree_obj,filtered_numb_dict,in_file_name);
        vector<string> mined;
        pattern_mine(tree_obj, support_count, mined);
        //mcout<<" level 1 "<< filtered_numb_dict.size()<<endl;

    }

  
    // std::cout << endl<<" TIME TAKEN "<<(get_time_spent_seconds());

/*****************************************************team code**********************************************************************************************/

/**************************************************************************************************************************************************************/
    // cout<<"toatl items in file : "<<total_items_in_file<<endl;

    return 0;


}

 
