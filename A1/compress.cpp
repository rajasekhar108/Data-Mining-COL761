#include<bits/stdc++.h>
#include <fstream>
#include <sstream>
#include <string>

int totalitems=0,total_items_in_data=0;
using namespace std;
unordered_set<string>final_encod_string;
string fp_fi,data_fi,dec_fi;
int max_lent=0;
int items_in_encoded_data=0;
unordered_set<int> get_nums_from_string(string s )
{
    unordered_set<int> no_in_each_str;
    istringstream iss(s);
    int token;
    while (iss >> token) {
        no_in_each_str.insert(token);
    }
    return no_in_each_str;
}

int lg(double argument)
{
    double base = 82.0;

    double result = log(argument) / log(base);
    int ceilValue = static_cast<int>(std::ceil(result));
    return ceilValue;
}

string generateRandomString(int length) {
    const std::string gen_from = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@#$%^&*()_-+={}[];':,./<>?";
    const int ch_len = gen_from.length();
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, ch_len - 1);
    string res;
    for (int i = 0; i < length; ++i) {
        res += gen_from[dis(gen)];
    }
    return res;
}


void givealllines(FILE *f,list<string>&alllines,int &c,int &no_fp)
{
    string line;

    while (true) {
        int ch = getc_unlocked(f);
        if (ch == EOF) {
            if (!line.empty()) {
                //c++;
                alllines.push_back(line);
            }
            break;
        } else if (ch == '\n') {
            c++;
            no_fp++;
            alllines.push_back(line);
            line.clear();
        } else {
            if(static_cast<char>(ch)==' ') c++;
            line += static_cast<char>(ch);
        }
    }
}

bool checkifsubset(unordered_set<int>&s1,unordered_set<int>&s2)
{
    for (int i : s1) {
        if (s2.find(i) == s2.end()) {
            return false;
        }
    }
    return true;
}

string encodeeachtxn(string &s,unordered_map<string,unordered_set<int>>&table)
{
    unordered_set<int> set_of_one_txn=get_nums_from_string(s);
    unordered_set<int> set_of_one_fp;
    vector<string> code_of_fp;
    total_items_in_data+=set_of_one_txn.size();
    for(auto i:table)
    {
        for(auto j: i.second)
        {
            set_of_one_fp.insert(j);
        }
        if(checkifsubset(set_of_one_fp,set_of_one_txn))
        {
            for(auto j: i.second)
            {
                set_of_one_txn.erase(j);
            }
            code_of_fp.push_back(i.first);
        }
        set_of_one_fp.clear();
    }
    items_in_encoded_data+=(set_of_one_txn.size()+code_of_fp.size());
    string encodec_txn="";
    for(auto i :code_of_fp)
    {
        encodec_txn+=i;
        encodec_txn+=" ";
        final_encod_string.insert(i);
    }
    
    
    for(auto i:set_of_one_txn) {encodec_txn+=to_string(i);encodec_txn+=" ";}
 
    return encodec_txn;
}


void d2(vector<unordered_set<int>> &topfps,int no_fp,int &no_fps_to_replace)
{
    vector<unordered_set<int>>tem;
    no_fps_to_replace=0;
    for(auto i: topfps)
    {
        int flag=1;
        for(int j=0;j<tem.size();j++)
        {
            if(checkifsubset(i,tem[j])){flag=0;break;}
        }
        if(flag) 
        {
            tem.push_back(i);
            no_fps_to_replace++;
        }
    }
    // cout<<"after subset no of fp : "<<no_fps_to_replace<<endl;
    topfps.clear();
    for(auto i:tem) topfps.push_back(i);
}

void dec_frq_patter(list<string> &all_lines_in_file,int per,vector<unordered_set<int>> &topfps,int no_fp,int &no_fps_to_replace)
{
    
    map<int,int>check;
    map<int,int>count;
    int max_len=0;
    for(auto i :all_lines_in_file)
    {
        count[get_nums_from_string(i).size()]++;;
        
    }
    int wa=0,sw=0;
    for(auto i: count)
    {
        max_len=max(max_len,i.first);
        // cout<<"length of items : "<<i.first<<"  no of items : "<<i.second<<endl;
        wa+=(i.first*i.second);
        sw+=i.second;
    }
    int wt_avg=wa/sw;
    int step_size=(90/(max_len-(wt_avg)));
    int k=0;
    map<int,int>limit;
    for(int i=max_len;i>wt_avg;i--)
    {
        limit[i]=((100-k*(step_size))*count[i]*0.01);
    }
    set<int>len;
    
    // cout<<"weighted avg : "<<wt_avg<<endl;
    int p=5;
    while(all_lines_in_file.size()>0)
    {
        unordered_set<int> temp1=get_nums_from_string(all_lines_in_file.front());
        if(temp1.size()<=wt_avg) {all_lines_in_file.pop_front();break; ;}

        if(check[temp1.size()]<(limit[temp1.size()]))
        {
            topfps.push_back(temp1);
            check[temp1.size()]++;
            no_fps_to_replace++;
        }
        all_lines_in_file.pop_front();
    }
    // cout<<"no fp after percentage : "<<no_fps_to_replace<<endl;
    d2(topfps,no_fp,no_fps_to_replace);



}
bool cmp(const std::string &str1, const std::string &str2) {
    return str1.length() > str2.length();
}

int main(int argc, char* argv[])
{
    fp_fi = argv[1];
    data_fi=argv[2];
    dec_fi=argv[3];

    FILE* inFile = fopen(fp_fi.c_str(), "r");
    list<string>all_lines_in_file;
    int all_numbers_in_fps=0,no_fp=0;
    givealllines(inFile,all_lines_in_file,all_numbers_in_fps,no_fp);
    all_lines_in_file.sort(cmp);

    fclose(inFile);
    vector<unordered_set<int>> topfps;
    int no_fps_to_replace=0;
    dec_frq_patter(all_lines_in_file,5,topfps,no_fp,no_fps_to_replace);
 
    all_lines_in_file.clear();
    // cout<<"vector created!!!!"<<endl;

    unordered_set<string> uniqueStrings;
    int len_of_each_encypt_code=lg(no_fps_to_replace)+1;
    // cout<<"no_fps_to_replace : "<<no_fps_to_replace<<endl<<"len_of_each_encypt_code : "<<len_of_each_encypt_code<<endl;
    while (uniqueStrings.size() < no_fps_to_replace) {
        string randomString = generateRandomString(len_of_each_encypt_code);
        uniqueStrings.insert(randomString);
    }
    // cout<<"uniqueStrings created!!"<<endl;

    unordered_map<string,unordered_set<int>>fp_mapping_table;
    auto it=uniqueStrings.begin();

    // cout<<"size of map : "<<topfps.size()<<endl;
   
    int total_items_in_mapping=0;
    for(auto i:topfps)
    {
        fp_mapping_table[*it]=i;it++;

    }
    topfps.clear();
    // cout<<"mapping completed!!!"<<endl;

    FILE* dat_file_ptr = fopen(data_fi.c_str(), "r");
    list<string>all_lines_in_data_file;
    int lines_in_data=0,all_items_in_datafile=0,no_of_txns=0;
    givealllines(dat_file_ptr,all_lines_in_data_file,all_items_in_datafile,no_of_txns);
    ofstream encoded(dec_fi);
    while(all_lines_in_data_file.size()>0)
    {
        string enc_str=encodeeachtxn(all_lines_in_data_file.front(),fp_mapping_table);
        all_lines_in_data_file.pop_front();
        encoded<<enc_str<<endl;;            
    }
    encoded<<"1mapping_starts1"<<endl;
    for(auto i:final_encod_string)
    {
        encoded<<i;
        for(auto j:fp_mapping_table[i]) {encoded<<" "<<j;}
        encoded<<endl;
        total_items_in_mapping+=(fp_mapping_table[i].size()+1);
    }
    // cout<<"total_items_in_mapping : "<<total_items_in_mapping<<endl;
    encoded.close();
    fp_mapping_table.clear();
    // cout<<"total_items_in_data : "<<total_items_in_data<<endl;
    // cout<<"items_in_encoded_data : "<<items_in_encoded_data<<endl;
    // cout<<"total after encoded : "<<(items_in_encoded_data+total_items_in_mapping)<<endl;;
    int size_of_encode=items_in_encoded_data+total_items_in_mapping;
    // cout<<"compression percentage : "<<(((total_items_in_data-size_of_encode)*(100.0)))/total_items_in_data<<endl;

}
