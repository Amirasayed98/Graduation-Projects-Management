<?php
      $connect = mysqli_connect("localhost", "root", "", "graduation");
     session_start();
      $cid2= $_SESSION['d']['cid2'] ;
    // echo $cid2;
     
?>
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Requirments Add </title>
    

    <link rel="stylesheet" href="css/bootstrap.min.css"/>
    
    <link rel="stylesheet" href="fontawesome/css/all.min.css">
    <link rel="stylesheet" href="css/19.css"/>
    <link rel="stylesheet" href="css/ai-cards-explore-style.css" >
    
    
    
</head>
<body>
<!-- start profile nav -->
<ul class="fixed-top nav justify-content-center my-profile-nav ">
      <li class="nav-item">
        <a class=" " href="#" tabindex="-1" aria-disabled="true"><i class="	fas fa-user-alt"></i> </a>
      </li>
      
      <li class="nav-item">
        
        <a class="nav-link nav-link1 " href="#">Explore</a>
      </li>
      <li class="nav-item">
        <a class="nav-link nav-link2" href="mycompanyreq.php">My Requirments</a>
      </li>
      <li class="nav-item">
        <a class="nav-link nav-link3 active" href="add_requriment.php">add new Reuirments</a>
      </li>
      <li class="nav-item">
        <a class="nav-link nav-link-my-logout " href="company.html" tabindex="-1" aria-disabled="true">Log Out</a>
      </li>
    </ul>
    
<!-- end profile nav -->

      <span id="error">

                     <?php
                     if(isset($_SESSION['errors'])){
                         if(!empty($_SESSION['errors'])){
                             echo '<div class="alert alert-danger">';
                             echo '<ul>';
                             foreach ($_SESSION['errors'] as $error){
                                 echo '<li>' ;
                                    echo $error;
                                 echo '</li>' ;
                             }
                             echo '</ul>';
                             echo   '</div>';
        }
        }
        ?>
        </span>
        
    <!--Start form-->
    <form action="save_add_req.php" method="post" class="shadow-lg main" enctype="multipart/form-data">
      
    
        <div class="container-fluid require_head">
            <h4>Your Requirments & Needs</h4>
        </div>

    <!--Strat Add new tracks-->

        <div class="require_body">
            <div class="tracks">
              <h2>Tracks</h2>
            </div>
            
            <hr class="col-md-12">
            <div >
           
           
            <select  class="custom-select" name="track"  id="validationCustom02" required >
                <option selected disabled value="">Choose Track...</option>
                <option value="Artficial Intelligence" <?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Artficial Intelligence') echo 'selected'?>>Artficial Intelligence</option>
                <option value="Virtual Reality"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Virtual Reality') echo 'selected'?>>Virtual Reality</option>
                <option value="Machine Learning" <?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Machine Learning') echo 'selected'?>>Machine Learning</option>
                <option value="Image Processing"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Image Processing') echo 'selected'?>>Image Processing</option>
                <option value="Internet Of Things"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Internet Of Things') echo 'selected'?>>Internet Of Things</option>
                <option value="Deep Learning"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Deep Learning') echo 'selected'?>>Deep Learning</option>
                <option value="Web Development"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Web Development') echo 'selected'?>>Web Development</option>
                <option value="Android Development"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Android Development') echo 'selected'?>>Android Development</option>
                <option value="Desktop Application"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Desktop Application') echo 'selected'?>>Desktop Application</option>
            </select>
            
          
        </div>
    <!--End Add new tracks-->

    <!--Strat Add new tools-->

       
     <!--End Add new tools-->

    <!--Strat Add new description-->

        <div class="require_body">
            <div class="tracks" id="short_desc">
                <h2>Short description ideas</h2>
            </div>
                
            <hr class="col-md-12">
            <div class="d1">
              <textarea class="form-control" class="input" id="input_description" name="idea" placeholder="Enter Short description..." rows="3" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['idea'] ?>"></textarea>
              <h6 class="short">Please write short, clear description (less than 6 sentences)</h6>
              <input type="hidden"   name="view" value=<?php echo $cid2; ?> />
              
            </div>
                    
        
        </div>
    <!--End Add new description-->

    <!--Strat footer-->
    
        <div class="container-fluid">
            <button type="button" class="btn btn-outline-secondary back " onclick="document.location ='ai-cards-explore.php'">Back</button>
            <button type="submit" class="btn btn-outline-secondary save ">Add</button>
        </div>
    
    <!--End footer-->
   
    
    </form>
    <!--End form-->

    
      <script src="js/jquery-3.4.1.min.js"></script>
      <script src="js/bootstrap.min.js"></script>

</body>
</html>
<?php
if(isset($_SESSION['errors'])){
    unset($_SESSION['errors']);
}
if(isset($_SESSION['data'])){
    unset($_SESSION['data']);
}
?>